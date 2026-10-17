using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.IO.MemoryMappedFiles;
using System.Linq;
using System.Reflection;
using System.Threading;
using System.Web.Script.Serialization;

namespace BeamicPro.Bridge
{
    // Vendor DLLs are loaded from the user's licensed BeamMic installation.
    // Reflection probes actual members rather than treating BeamGage examples as guarantees.
    internal sealed class DeviceProfile
    {
        public string beamMicDirectory { get; set; } = @"C:\Program Files\Spiricon\BeamMic";
        public string instanceName { get; set; } = "BeamicPro";
        public string framePath { get; set; } = "ResultsPriorityFrame";
        public string imagePath { get; set; } = "ResultsPriorityFrame.DoubleData";
        public string widthPath { get; set; } = "FrameInfoResults.Width";
        public string heightPath { get; set; } = "FrameInfoResults.Height";
        public string idPath { get; set; } = "FrameInfoResults.ID";
        public string frameExposurePath { get; set; } = "FrameInfoResults.Exposure";
        public string[] coordinatePaths { get; set; } = new[] {
            "FrameInfoResults.Width", "FrameInfoResults.Height", "FrameInfoResults.OffsetX",
            "FrameInfoResults.OffsetY", "FrameInfoResults.BinningX", "FrameInfoResults.BinningY",
            "FrameInfoResults.ScaleMultiplier", "FrameInfoResults.Gamma", "FrameInfoResults.BitsPerPixel",
            "FrameInfoResults.Gain", "FrameInfoResults.BlackLevel" };
        public double fullScale { get; set; } = 0;
        public bool saturationVerified { get; set; } = false;
        public bool frameExposureVerified { get; set; } = false;
        public string verifiedForAssemblyVersion { get; set; } = "";
        public string verifiedForSource { get; set; } = "";
        public int verifiedBitsPerPixel { get; set; } = 0;
        public double frameExposureToUs { get; set; } = 0;
        public double pixelPitchUm { get; set; } = 0;
        public string processingDescription { get; set; } = "unverified";
    }

    internal sealed class Bridge : IDisposable
    {
        private readonly DeviceProfile profile;
        private readonly JavaScriptSerializer json = new JavaScriptSerializer();
        private readonly object outputLock = new object();
        private readonly object sdkLock = new object();
        private readonly AutoResetEvent newFrame = new AutoResetEvent(false);
        private readonly TextWriter output;
        private object app, frameEvents;
        private Delegate frameDelegate;
        private Thread reader;
        private volatile bool stopped;
        private volatile bool paused = true;
        private double fps = 20, unitToUs;
        private long epoch, lastVendorId = -1, sequence;
        private string verifiedVersion;
        private MemoryMappedFile map;
        private MemoryMappedViewAccessor view;
        private string mapName;
        private int capacity, slot = -1;
        private long slotSpan;
        private readonly List<MemoryMappedFile> oldMaps = new List<MemoryMappedFile>();
        private readonly List<MemoryMappedViewAccessor> oldViews = new List<MemoryMappedViewAccessor>();
        private bool exposureControl;
        private bool manualExposureOwned;
        private bool previousAutoX;
        private bool canRestoreAutoX;
        private double originalExposure;
        private int consecutiveErrors;
        private Dictionary<string, object> capabilities;
        private string originalSource;
        private bool numericReported;

        public Bridge(DeviceProfile profile, TextWriter output) { this.profile = profile; this.output = output; }

        public void Emit(object value)
        {
            lock (outputLock) { output.WriteLine(json.Serialize(value)); output.Flush(); }
        }

        private object PathValue(string path)
        {
            object value = app;
            foreach (var name in path.Split('.'))
            {
                if (value == null) throw new InvalidOperationException("Null at " + path);
                var type = value.GetType();
                var property = type.GetProperty(name) ?? type.GetInterfaces().Select(i => i.GetProperty(name)).FirstOrDefault(p => p != null);
                if (property == null) throw new MissingMemberException(type.FullName, name);
                value = property.GetValue(value, null);
            }
            return value;
        }

        private object Invoke(object target, string name, params object[] arguments)
        {
            var method = target.GetType().GetMethods().Concat(target.GetType().GetInterfaces().SelectMany(i => i.GetMethods()))
                .FirstOrDefault(m => m.Name == name && m.GetParameters().Length == arguments.Length);
            if (method == null) throw new MissingMethodException(target.GetType().FullName, name);
            return method.Invoke(target, arguments);
        }

        private object Egb(string method, double? value = null)
        {
            var target = PathValue("EGB");
            var member = target.GetType().GetMethods().Concat(target.GetType().GetInterfaces().SelectMany(i => i.GetMethods()))
                .FirstOrDefault(m => m.Name == method && m.GetParameters().Length == (value.HasValue ? 2 : 1)
                                  && m.GetParameters()[0].ParameterType.IsEnum);
            if (member == null) throw new MissingMethodException("EGB", method);
            var exposure = Enum.Parse(member.GetParameters()[0].ParameterType, "EXPOSURE", true);
            return member.Invoke(target, value.HasValue ? new object[] { exposure, value.Value } : new[] { exposure });
        }

        private static double Number(object value) => Convert.ToDouble(value, CultureInfo.InvariantCulture);
        private bool Verified {
            get {
                try {
                    return profile.verifiedForAssemblyVersion == verifiedVersion && !String.IsNullOrEmpty(verifiedVersion)
                        && !String.IsNullOrEmpty(profile.verifiedForSource)
                        && profile.verifiedForSource == Convert.ToString(PathValue("DataSource.DataSource"), CultureInfo.InvariantCulture)
                        && profile.verifiedBitsPerPixel > 0
                        && Convert.ToInt32(PathValue("FrameInfoResults.BitsPerPixel")) == profile.verifiedBitsPerPixel
                        && Math.Abs(Number(PathValue("FrameInfoResults.Gamma")) - 1) < 0.000001;
                } catch { return false; }
            }
        }
        private static bool Finite(double value) => !Double.IsNaN(value) && !Double.IsInfinity(value);
        private static double UnitMultiplier(string value)
        {
            switch (value.Trim().ToLowerInvariant())
            {
                case "us": case "µs": case "μs": case "microseconds": return 1;
                case "ms": case "milliseconds": return 1000;
                case "s": case "seconds": return 1000000;
                default: throw new InvalidOperationException("Unverified exposure unit: " + value);
            }
        }

        public void Start()
        {
            var directory = profile.beamMicDirectory;
            if (!Directory.Exists(directory)) throw new DirectoryNotFoundException("BeamMic installation not found: " + directory);
            AppDomain.CurrentDomain.AssemblyResolve += (sender, args) => {
                var path = System.IO.Path.Combine(directory, new AssemblyName(args.Name).Name + ".dll");
                return File.Exists(path) ? Assembly.LoadFrom(path) : null;
            };
            var assemblies = new List<Assembly>();
            foreach (var name in new[] { "Spiricon.Automation.dll", "Spiricon.BeamMic.Automation.dll" })
                assemblies.Add(Assembly.LoadFrom(System.IO.Path.Combine(directory, name)));
            verifiedVersion = assemblies[1].GetName().Version.ToString();
            var type = assemblies.SelectMany(a => a.GetTypes()).FirstOrDefault(t => t.Name == "AutomatedBeamMic");
            if (type == null) throw new TypeLoadException("AutomatedBeamMic is not exposed by these assemblies");
            app = Activator.CreateInstance(type, profile.instanceName, true);
            var eventType = assemblies.SelectMany(a => a.GetTypes()).FirstOrDefault(t => t.Name == "AutomationFrameEvents");
            if (eventType == null) throw new TypeLoadException("AutomationFrameEvents is unavailable");
            frameEvents = Activator.CreateInstance(eventType, PathValue(profile.framePath));
            var eventInfo = eventType.GetEvent("OnNewFrame");
            if (eventInfo == null) throw new MissingMemberException("OnNewFrame");
            frameDelegate = Delegate.CreateDelegate(eventInfo.EventHandlerType, this,
                GetType().GetMethod(nameof(FrameReady), BindingFlags.NonPublic | BindingFlags.Instance));
            eventInfo.AddEventHandler(frameEvents, frameDelegate);
            var report = new Dictionary<string, object> {
                ["simulated"] = false, ["assembly_version"] = verifiedVersion,
                ["source"] = SafeValue("DataSource.DataSource"), ["frame_path"] = profile.framePath,
                ["profile_verified"] = Verified, ["processing"] = profile.processingDescription,
                ["saturation_verified"] = Verified && profile.saturationVerified && profile.fullScale > 0,
                ["exposure_verified"] = Verified && profile.frameExposureVerified && profile.frameExposureToUs > 0,
                ["numeric_images"] = false, ["frame_ids"] = false, ["ready"] = false,
                ["members"] = DiscoverMembers(), ["note"] = "Frame fields are runtime-probed. No camera throughput guarantee."
            };
            try
            {
                unitToUs = UnitMultiplier(Convert.ToString(Egb("Units"), CultureInfo.InvariantCulture));
                report["minimum_us"] = Number(Egb("RangeMin")) * unitToUs;
                report["maximum_us"] = Number(Egb("RangeMax")) * unitToUs;
                report["step_us"] = Math.Max(Number(Egb("Increment")) * unitToUs, 0.000001);
                originalExposure = Number(Egb("Get"));
                var egbType = PathValue("EGB").GetType();
                exposureControl = egbType.GetMethods().Concat(egbType.GetInterfaces().SelectMany(i => i.GetMethods()))
                    .Any(m => m.Name == "Set" && m.GetParameters().Length == 2);
                try { previousAutoX = Convert.ToBoolean(PathValue("Calibration.AutoXIsEnabled")); canRestoreAutoX = true; }
                catch { exposureControl = false; report["exposure_error"] = "AutoX state cannot be read; manual exposure ownership is unverified"; }
            }
            catch (Exception error) { report["exposure_error"] = error.GetBaseException().Message; }
            report["exposure_control"] = exposureControl;
            report["exposure_owned"] = manualExposureOwned;
            capabilities = report;
            Emit(new { type = "capabilities", report });
            reader = new Thread(ReadFrames) { IsBackground = true };
            reader.Start();
            // User selects their camera in the original application. Starting capture is explicit from the host.
        }

        private object SafeValue(string path) { try { return Convert.ToString(PathValue(path), CultureInfo.InvariantCulture); } catch { return null; } }
        private Dictionary<string, string[]> DiscoverMembers()
        {
            var result = new Dictionary<string, string[]>();
            foreach (var name in new[] { "", profile.framePath, "FrameInfoResults", "EGB", "Calibration", "DataSource" })
            {
                try {
                    var target = name == "" ? app : PathValue(name);
                    result[name] = target.GetType().GetProperties().Concat(target.GetType().GetInterfaces().SelectMany(t => t.GetProperties()))
                        .Select(p => p.Name).Distinct().OrderBy(n => n).ToArray();
                } catch { result[name] = new string[0]; }
            }
            return result;
        }
        private void FrameReady() { if (!stopped && !paused) newFrame.Set(); }

        private void ReadFrames()
        {
            var clock = Stopwatch.StartNew();
            double lastRead = -100;
            while (!stopped)
            {
                if (!newFrame.WaitOne(500) || paused) continue;
                var sleep = 1 / fps - (clock.Elapsed.TotalSeconds - lastRead);
                if (sleep > 0) Thread.Sleep((int)(sleep * 1000));
                if (stopped || paused) continue;
                try
                {
                    Dictionary<string, object> descriptor;
                    lock (sdkLock)
                    {
                        var before = Convert.ToInt64(PathValue(profile.idPath));
                        if (before == lastVendorId) continue;
                        int width = Convert.ToInt32(PathValue(profile.widthPath)), height = Convert.ToInt32(PathValue(profile.heightPath));
                        if (width <= 0 || height <= 0 || (long)width * height > 50000000)
                            throw new InvalidOperationException("Invalid numeric image dimensions");
                        var array = PathValue(profile.imagePath) as Array;
                        if (array == null || array.Rank != 1 || array.Length != width * height)
                            throw new InvalidOperationException("Image member is not a matching numeric raster");
                        var values = new float[array.Length];
                        if (array is double[] doubles) {
                            for (int i = 0; i < doubles.Length; i++) {
                                if (!Finite(doubles[i])) throw new InvalidOperationException("Nonfinite pixels");
                                values[i] = (float)doubles[i];
                            }
                        } else if (array is float[] floats) {
                            Array.Copy(floats, values, values.Length);
                            if (values.Any(v => !Finite(v))) throw new InvalidOperationException("Nonfinite pixels");
                        } else {
                            for (int i = 0; i < array.Length; i++) {
                                double value = Number(array.GetValue(i));
                                if (!Finite(value)) throw new InvalidOperationException("Nonfinite pixels");
                                values[i] = (float)value;
                            }
                        }
                        var coordinate = profile.coordinatePaths.Select(p => p + "=" + Convert.ToString(PathValue(p), CultureInfo.InvariantCulture)).ToArray();
                        double exposureUs = Number(PathValue(profile.frameExposurePath)) * profile.frameExposureToUs;
                        var after = Convert.ToInt64(PathValue(profile.idPath));
                        if (before != after) continue;
                        if (!Finite(exposureUs) || exposureUs <= 0 || profile.fullScale <= 0)
                            throw new InvalidOperationException("Exposure association or saturation full scale is unconfigured");
                        if (map == null || values.Length > capacity) Allocate(values.Length);
                        slot = (slot + 1) % 3;
                        long offset = slot * slotSpan;
                        sequence += 2;
                        view.Write(offset, sequence - 1);
                        Thread.MemoryBarrier();
                        view.WriteArray(offset + 8, values, 0, values.Length);
                        Thread.MemoryBarrier();
                        view.Write(offset, sequence);
                        lastVendorId = after;
                        descriptor = new Dictionary<string, object> {
                            ["map_name"] = mapName, ["map_size"] = 3L * slotSpan,
                            ["offset"] = offset, ["sequence"] = sequence, ["shape"] = new[] { height, width },
                            ["frame"] = new Dictionary<string, object> {
                                ["frame_id"] = sequence / 2, ["received_ns"] = 0,
                                ["exposure_us"] = exposureUs, ["exposure_epoch"] = epoch,
                                ["coordinate_key"] = String.Join("|", coordinate), ["full_scale"] = profile.fullScale,
                                ["pixel_pitch_um"] = profile.pixelPitchUm > 0 ? (object)profile.pixelPitchUm : null,
                                ["timestamp_source"] = "host_receive_only", ["exposure_verified"] = Verified && profile.frameExposureVerified,
                                ["saturation_verified"] = Verified && profile.saturationVerified,
                                ["metadata"] = new { assembly_version = verifiedVersion, processing = profile.processingDescription, vendor_frame_id = after, simulated = false }
                            }
                        };
                    }
                    lastRead = clock.Elapsed.TotalSeconds;
                    consecutiveErrors = 0;
                    if (!numericReported) {
                        numericReported = true;
                        capabilities["numeric_images"] = true;
                        capabilities["frame_ids"] = true;
                        capabilities["source"] = SafeValue("DataSource.DataSource");
                        capabilities["profile_verified"] = Verified;
                        capabilities["saturation_verified"] = Verified && profile.saturationVerified;
                        capabilities["exposure_verified"] = Verified && profile.frameExposureVerified;
                        capabilities["ready"] = Verified && profile.saturationVerified && profile.frameExposureVerified && exposureControl;
                        Emit(new { type = "capabilities", report = capabilities });
                    }
                    Emit(new { type = "frame", descriptor });
                }
                catch (Exception error)
                {
                    consecutiveErrors++;
                    if (consecutiveErrors == 1 || consecutiveErrors % 20 == 0)
                        Emit(new { type = "error", message = error.GetBaseException().Message, fatal = false });
                }
            }
        }

        private void Allocate(int pixels)
        {
            if (map != null) { view.Dispose(); map.Dispose(); }
            capacity = pixels;
            slotSpan = 8L + ((pixels * 4L + 7) / 8) * 8;
            mapName = "Local\\BeamicPro_" + Guid.NewGuid().ToString("N");
            map = MemoryMappedFile.CreateNew(mapName, 3L * slotSpan);
            view = map.CreateViewAccessor();
            slot = -1;
        }

        public object Command(Dictionary<string, object> command)
        {
            string method = Convert.ToString(command["method"]);
            if (method == "set_rate") {
                double value = Number(command["fps"]);
                if (!Finite(value) || value < 1 || value > 500) throw new ArgumentOutOfRangeException("fps");
                fps = value;
                return new { fps, controls = "bridge_fetch_rate", camera_rate_control = false };
            }
            if (method == "shutdown") { stopped = true; newFrame.Set(); return new { stopped = true }; }
            lock (sdkLock)
            {
                if (method == "probe") {
                    try {
                        unitToUs = UnitMultiplier(Convert.ToString(Egb("Units"), CultureInfo.InvariantCulture));
                        capabilities["minimum_us"] = Number(Egb("RangeMin")) * unitToUs;
                        capabilities["maximum_us"] = Number(Egb("RangeMax")) * unitToUs;
                        capabilities["step_us"] = Math.Max(Number(Egb("Increment")) * unitToUs, 0.000001);
                        if (!manualExposureOwned) {
                            originalExposure = Number(Egb("Get"));
                            previousAutoX = Convert.ToBoolean(PathValue("Calibration.AutoXIsEnabled"));
                        }
                        canRestoreAutoX = true;
                        var egbType = PathValue("EGB").GetType();
                        exposureControl = egbType.GetMethods().Concat(egbType.GetInterfaces().SelectMany(i => i.GetMethods()))
                            .Any(m => m.Name == "Set" && m.GetParameters().Length == 2);
                    } catch (Exception error) { exposureControl = false; capabilities["exposure_error"] = error.GetBaseException().Message; }
                    capabilities["exposure_control"] = exposureControl;
                    capabilities["exposure_owned"] = manualExposureOwned;
                    capabilities["source"] = SafeValue("DataSource.DataSource");
                    capabilities["profile_verified"] = Verified;
                    capabilities["saturation_verified"] = Verified && profile.saturationVerified;
                    capabilities["exposure_verified"] = Verified && profile.frameExposureVerified;
                    capabilities["members"] = DiscoverMembers();
                    Emit(new { type = "capabilities", report = capabilities });
                    return capabilities;
                }
                if (method == "claim_exposure") {
                    if (!Verified || !profile.frameExposureVerified || !profile.saturationVerified || !exposureControl)
                        throw new InvalidOperationException("Exposure ownership requires a verified device profile and a selected camera");
                    if (!manualExposureOwned) {
                        originalExposure = Number(Egb("Get"));
                        originalSource = SafeValue("DataSource.DataSource") as string;
                        previousAutoX = Convert.ToBoolean(PathValue("Calibration.AutoXIsEnabled"));
                    }
                    if (Convert.ToBoolean(PathValue("Calibration.AutoXIsEnabled"))) Invoke(PathValue("Calibration"), "AutoX");
                    manualExposureOwned = true;
                    capabilities["exposure_owned"] = true;
                    Emit(new { type = "capabilities", report = capabilities });
                    return new { exposure_owned = true };
                }
                if (method == "start") {
                    if (Verified && profile.frameExposureVerified && profile.saturationVerified && exposureControl) {
                        if (!manualExposureOwned) {
                            originalExposure = Number(Egb("Get"));
                            originalSource = SafeValue("DataSource.DataSource") as string;
                            previousAutoX = Convert.ToBoolean(PathValue("Calibration.AutoXIsEnabled"));
                        }
                        if (Convert.ToBoolean(PathValue("Calibration.AutoXIsEnabled"))) Invoke(PathValue("Calibration"), "AutoX");
                        manualExposureOwned = true;
                        capabilities["exposure_owned"] = true;
                    }
                    paused = false;
                    try { Invoke(PathValue("DataSource"), "Start"); } catch { paused = true; throw; }
                    return new { started = true };
                }
                if (method == "stop") { paused = true; Invoke(PathValue("DataSource"), "Stop"); return new { stopped = true }; }
                if (method == "set_exposure")
                {
                    if (!Verified || !profile.frameExposureVerified || !profile.saturationVerified || !exposureControl)
                        throw new InvalidOperationException("Device profile has not passed metrology verification");
                    double requested = Number(command["exposure_us"]);
                    if (!Finite(requested) || requested <= 0) throw new ArgumentOutOfRangeException("exposure_us");
                    double native = requested / unitToUs;
                    double low = Number(Egb("RangeMin")), high = Number(Egb("RangeMax")), step = Number(Egb("Increment"));
                    if (step > 0) native = low + Math.Round((native - low) / step) * step;
                    native = Math.Max(low, Math.Min(high, native));
                    if (native <= 0) throw new InvalidOperationException("No positive exposure is representable at this increment");
                    if (!manualExposureOwned) {
                        originalExposure = Number(Egb("Get"));
                        originalSource = SafeValue("DataSource.DataSource") as string;
                        previousAutoX = Convert.ToBoolean(PathValue("Calibration.AutoXIsEnabled"));
                    }
                    if (Convert.ToBoolean(PathValue("Calibration.AutoXIsEnabled"))) Invoke(PathValue("Calibration"), "AutoX");
                    manualExposureOwned = true;
                    Egb("Set", native);
                    epoch++;
                    return new { exposure_us = Number(Egb("Get")) * unitToUs, epoch };
                }
            }
            throw new InvalidOperationException("Unsupported method: " + method);
        }

        public void Dispose()
        {
            stopped = true;
            newFrame.Set();
            reader?.Join(1500);
            lock (sdkLock)
            {
                if (app != null) {
                    try {
                        if (manualExposureOwned && Equals(originalSource, SafeValue("DataSource.DataSource"))) Egb("Set", originalExposure);
                        if (canRestoreAutoX && Equals(originalSource, SafeValue("DataSource.DataSource")) && Convert.ToBoolean(PathValue("Calibration.AutoXIsEnabled")) != previousAutoX)
                            Invoke(PathValue("Calibration"), "AutoX");
                    } catch (Exception error) { Console.Error.WriteLine("Restore settings: " + error.GetBaseException().Message); }
                    // This bridge owns the named BeamMic instance it created.
                    try { Invoke(PathValue("Instance"), "Shutdown"); } catch { }
                    if (app is IDisposable disposable) disposable.Dispose();
                }
            }
            view?.Dispose(); map?.Dispose();
            foreach (var item in oldViews) item.Dispose();
            foreach (var item in oldMaps) item.Dispose();
            newFrame.Dispose();
        }
    }

    internal static class Program
    {
        private static int Main(string[] args)
        {
            Console.OutputEncoding = new System.Text.UTF8Encoding(false);
            Console.InputEncoding = new System.Text.UTF8Encoding(false);
            var protocol = Console.Out;
            Console.SetOut(Console.Error);
            var json = new JavaScriptSerializer();
            Bridge bridge = null;
            try
            {
                var profile = new DeviceProfile();
                int index = Array.IndexOf(args, "--profile");
                if (index >= 0) profile = json.Deserialize<DeviceProfile>(File.ReadAllText(args[index + 1]));
                bridge = new Bridge(profile, protocol);
                bridge.Start();
                string line;
                while ((line = Console.ReadLine()) != null)
                {
                    var command = json.Deserialize<Dictionary<string, object>>(line);
                    object id = command.ContainsKey("id") ? command["id"] : 0;
                    try {
                        var result = bridge.Command(command);
                        bridge.Emit(new { type = "reply", id, ok = true, result });
                        if (Convert.ToString(command["method"]) == "shutdown") break;
                    } catch (Exception error) { bridge.Emit(new { type = "reply", id, ok = false, error = error.GetBaseException().Message }); }
                }
                return 0;
            }
            catch (Exception error)
            {
                lock (protocol) { protocol.WriteLine(json.Serialize(new { type = "error", fatal = true, message = error.GetBaseException().Message })); protocol.Flush(); }
                return 1;
            }
            finally { bridge?.Dispose(); }
        }
    }
}

