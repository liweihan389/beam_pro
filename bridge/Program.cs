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
        public string widthPath { get; set; } = "ResultsPriorityFrame.Width";
        public string heightPath { get; set; } = "ResultsPriorityFrame.Height";
        public string idPath { get; set; } = "ResultsPriorityFrame.UniqueKey";
        // Unknown exposure/processing metadata must not prevent numerical preview.
        public string frameExposurePath { get; set; } = "";
        public string bitsPerPixelPath { get; set; } = "ResultsPriorityFrame.OriginalBpp";
        public string gammaPath { get; set; } = "";
        public bool frameIdentityVerified { get; set; } = false;
        public bool callbackSnapshotVerified { get; set; } = false;
        public bool liveSourceVerified { get; set; } = false;
        public bool threadModelVerified { get; set; } = false;
        public string[] coordinatePaths { get; set; } = new[] {
            "ResultsPriorityFrame.Width", "ResultsPriorityFrame.Height", "ResultsPriorityFrame.OriginalBpp",
            "ResultsPriorityFrame.IsBaselineCalibrated", "ResultsPriorityFrame.IsFrameCalibrated" };
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
        private readonly SnapshotGate gate = new SnapshotGate();
        private readonly SnapshotMailbox mailbox = new SnapshotMailbox();
        private readonly object reportLock = new object();
        private readonly AutoResetEvent newFrame = new AutoResetEvent(false);
        private readonly TextWriter output;
        private object app, frameEvents;
        private Delegate frameDelegate;
        private Thread reader;
        private volatile bool stopped;
        private volatile bool paused = true;
        private double fps = 20, unitToUs;
        private long epoch, sequence;
        private string lastVendorId;
        private readonly Queue<object> frameSamples = new Queue<object>();
        private long frameEventCount, duplicateFrames;
        private long controlGeneration, acceptedSnapshots, rateSkipped, gateSkipped, snapshotErrors, nextAcceptTicks;
        private string pendingError;
        private double? settingExposureUs;
        private string settingReadUtc;
        private double lastCallbackMs, lastCopyMs, lastPublishMs;
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
            return VendorAccess.Value(app, path);
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
                        && profile.callbackSnapshotVerified && profile.liveSourceVerified && profile.threadModelVerified
                        && Convert.ToInt32(PathValue(profile.bitsPerPixelPath)) == profile.verifiedBitsPerPixel
                        && !String.IsNullOrEmpty(profile.gammaPath)
                        && Math.Abs(Number(PathValue(profile.gammaPath)) - 1) < 0.000001;
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
            // Before event subscription, establish a stable state for discovery.
            Invoke(PathValue("DataSource"), "Stop");
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
                ["members"] = DiscoverMembers(), ["diagnostics"] = Diagnostics(),
                ["preview_only"] = true,
                ["note"] = "Snapshots are owned within callbacks. Software sequences are not hardware frame IDs; live provenance, callback and exposure contracts still require verification."
            };
            try
            {
                unitToUs = UnitMultiplier(Convert.ToString(Egb("Units"), CultureInfo.InvariantCulture));
                report["minimum_us"] = Number(Egb("RangeMin")) * unitToUs;
                report["maximum_us"] = Number(Egb("RangeMax")) * unitToUs;
                report["step_us"] = Math.Max(Number(Egb("Increment")) * unitToUs, 0.000001);
                originalExposure = Number(Egb("Get"));
                settingExposureUs = originalExposure * unitToUs;
                settingReadUtc = DateTime.UtcNow.ToString("o");
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
            foreach (var name in new[] { "", profile.framePath, "FrameInfoResults", "FrameBuffer", "EGB", "Calibration", "DataSource",
                "ResultsPriorityFrame.FrameData", "ResultsPriorityFrame.CalibratedData", "ResultsPriorityFrame.Calibration", "OpticalScale" })
            {
                try {
                    var target = name == "" ? app : PathValue(name);
                    result[name] = target.GetType().GetProperties().Concat(target.GetType().GetInterfaces().SelectMany(t => t.GetProperties()))
                        .Select(p => p.Name).Distinct().OrderBy(n => n).ToArray();
                } catch { result[name] = new string[0]; }
            }
            return result;
        }
        private object Diagnostics()
        {
            var paths = new[] { profile.framePath, profile.imagePath, profile.widthPath, profile.heightPath,
                profile.idPath, profile.frameExposurePath, profile.bitsPerPixelPath, profile.gammaPath,
                "ResultsPriorityFrame.HasData", "ResultsPriorityFrame.CalibratedData", "ResultsPriorityFrame.FrameData", "ResultsPriorityFrame.BaseLineOffset",
                "Calibration.Status", "DataSource.Status" }.Concat(profile.coordinatePaths ?? new string[0]);
            return new {
                fields = VendorAccess.Describe(app, paths),
                methods = VendorAccess.Methods(app, new[] { profile.framePath, "FrameBuffer", "EGB", "DataSource", "Calibration" }),
                new_frame_events = Interlocked.Read(ref frameEventCount),
                duplicate_markers = duplicateFrames,
                snapshot_consistency = "callback ownership requires installed-version verification; UniqueKey is not an atomicity check",
                accepted_snapshots = acceptedSnapshots, rate_skipped = rateSkipped, gate_skipped = gateSkipped,
                replaced_pending = mailbox.Replaced, snapshot_errors = snapshotErrors,
                callback_ms = lastCallbackMs, copy_ms = lastCopyMs, publish_ms = lastPublishMs,
                control_generation = controlGeneration,
                probe_scope = "Frame getter diagnostics are collected with the data source stopped; a running source is resumed afterward",
                frame_samples = frameSamples.ToArray(), images_saved = false,
                missing_requirements = new[] {
                    !(profile.callbackSnapshotVerified && profile.liveSourceVerified && profile.threadModelVerified) ? "Callback consistency, live-frame provenance and SDK thread model unverified" : null,
                    !(Verified && profile.frameExposureVerified && profile.frameExposureToUs > 0) ? "Frame-associated exposure unverified" : null,
                    !(Verified && profile.saturationVerified && profile.fullScale > 0) ? "Linear image stage and saturation scale unverified" : null
                }.Where(v => v != null).ToArray()
            };
        }
        private object FrameValue(object frame, string path) {
            string prefix = profile.framePath + ".";
            return path.StartsWith(prefix, StringComparison.Ordinal) ? VendorAccess.Value(frame, path.Substring(prefix.Length)) : PathValue(path);
        }

        private void FrameReady()
        {
            long eventId = Interlocked.Increment(ref frameEventCount);
            if (stopped || paused || !gate.TryEnter()) { Interlocked.Increment(ref gateSkipped); return; }
            long started = Stopwatch.GetTimestamp();
            try {
                if (started < Interlocked.Read(ref nextAcceptTicks)) { Interlocked.Increment(ref rateSkipped); return; }
                Interlocked.Exchange(ref nextAcceptTicks, started + (long)(Stopwatch.Frequency / fps));
                long generation = Interlocked.Read(ref controlGeneration);
                // Read all frame fields from one captured object, before returning to the vendor.
                object frame = PathValue(profile.framePath);
                int width = Convert.ToInt32(FrameValue(frame, profile.widthPath));
                int height = Convert.ToInt32(FrameValue(frame, profile.heightPath));
                Array vendorPixels = FrameValue(frame, profile.imagePath) as Array;
                long copyStarted = Stopwatch.GetTimestamp();
                Array owned = NativeRaster.Own(vendorPixels, width, height);
                double copyMs = NativeRaster.Ms(Stopwatch.GetTimestamp() - copyStarted);
                string marker = null;
                try { marker = VendorAccess.Token(FrameValue(frame, profile.idPath)); } catch { }
                if (marker != null && marker == lastVendorId) Interlocked.Increment(ref duplicateFrames);
                lastVendorId = marker; // Diagnostic only. Identical pixels/keys do not suppress events.
                string source = Convert.ToString(PathValue("DataSource.DataSource"), CultureInfo.InvariantCulture);
                string status = Convert.ToString(PathValue("DataSource.Status"), CultureInfo.InvariantCulture);
                bool complete = true;
                var coordinates = new List<string> { "source=" + source, "image=" + profile.imagePath, "width=" + width, "height=" + height };
                foreach (string path in profile.coordinatePaths ?? new string[0]) {
                    try { coordinates.Add(path + "=" + Convert.ToString(FrameValue(frame, path), CultureInfo.InvariantCulture)); }
                    catch { complete = false; coordinates.Add(path + "=?"); }
                }
                double? exposure = null;
                if (!String.IsNullOrWhiteSpace(profile.frameExposurePath) && profile.frameExposureToUs > 0) {
                    try {
                        double value = Number(FrameValue(frame, profile.frameExposurePath)) * profile.frameExposureToUs;
                        if (Finite(value) && value > 0) exposure = value;
                    } catch { }
                }
                bool verified = Verified;
                bool quantitative = verified && complete && String.Equals(status, "RUNNING", StringComparison.OrdinalIgnoreCase)
                    && exposure.HasValue && profile.frameExposureVerified && profile.saturationVerified
                    && Finite(profile.fullScale) && profile.fullScale > 0;
                long identity = Interlocked.Increment(ref acceptedSnapshots);
                double callbackMs = NativeRaster.Ms(Stopwatch.GetTimestamp() - started);
                var snapshot = new NativeSnapshot {
                    Pixels = owned, Width = width, Height = height, EventId = eventId, SnapshotId = identity,
                    Generation = generation, CopyMs = copyMs, CallbackMs = callbackMs, Marker = marker,
                    Source = source, Status = status, Coordinate = String.Join("|", coordinates),
                    Quantitative = quantitative, ExposureUs = exposure, ExposureEpoch = epoch,
                    Metadata = new Dictionary<string, object> {
                        ["assembly_version"] = verifiedVersion, ["processing"] = profile.processingDescription,
                        ["vendor_frame_id"] = marker, ["event_id"] = eventId, ["snapshot_id"] = identity,
                        ["control_generation"] = generation, ["source"] = source, ["source_status"] = status,
                        ["frame_identity_verified"] = quantitative, ["software_sequence_is_hardware_id"] = false,
                        ["provenance"] = quantitative ? "verified_live_callback" : "unverified_vendor_callback",
                        ["callback_ms"] = callbackMs, ["copy_ms"] = copyMs,
                        ["exposure_setting_us"] = settingExposureUs, ["exposure_setting_read_utc"] = settingReadUtc,
                        ["exposure_setting_source"] = "last_control_query; not frame-associated exposure",
                        ["preview_only"] = !quantitative, ["simulated"] = false
                    }
                };
                if (generation != Interlocked.Read(ref controlGeneration) || stopped || paused) return;
                mailbox.Put(snapshot);
                newFrame.Set();
            } catch (Exception error) {
                Interlocked.Increment(ref snapshotErrors);
                Interlocked.Exchange(ref pendingError, error.GetBaseException().Message);
                newFrame.Set(); // Only the worker reports errors; never write pipes from a callback.
            } finally { gate.Leave(); }
        }

        private void ReadFrames()
        {
            while (!stopped) {
                newFrame.WaitOne(500);
                string error = Interlocked.Exchange(ref pendingError, null);
                if (error != null) Emit(new { type = "error", message = error, fatal = false });
                NativeSnapshot snapshot = mailbox.Take();
                if (snapshot == null || snapshot.Generation != Interlocked.Read(ref controlGeneration) || paused) continue;
                try {
                    long started = Stopwatch.GetTimestamp();
                    double[] range = NativeRaster.Range(snapshot.Pixels);
                    byte[] bytes = new byte[Buffer.ByteLength(snapshot.Pixels)];
                    Buffer.BlockCopy(snapshot.Pixels, 0, bytes, 0, bytes.Length);
                    if (map == null || bytes.Length > capacity) Allocate(bytes.Length);
                    slot = (slot + 1) % 3;
                    long offset = slot * slotSpan;
                    sequence += 2;
                    view.Write(offset, sequence - 1);
                    Thread.MemoryBarrier();
                    view.WriteArray(offset + 8, bytes, 0, bytes.Length);
                    Thread.MemoryBarrier();
                    view.Write(offset, sequence);
                    snapshot.Metadata["wire_dtype"] = NativeRaster.Dtype(snapshot.Pixels.GetType().GetElementType());
                    snapshot.Metadata["publish_ms"] = NativeRaster.Ms(Stopwatch.GetTimestamp() - started);
                    var descriptor = new Dictionary<string, object> {
                        ["map_name"] = mapName, ["map_size"] = 3L * slotSpan,
                        ["offset"] = offset, ["sequence"] = sequence, ["shape"] = new[] { snapshot.Height, snapshot.Width },
                        ["dtype"] = snapshot.Metadata["wire_dtype"], ["control_generation"] = snapshot.Generation,
                        ["preview_only"] = !snapshot.Quantitative,
                        ["frame"] = new Dictionary<string, object> {
                            ["frame_id"] = snapshot.SnapshotId, ["received_ns"] = 0,
                            ["exposure_us"] = snapshot.Quantitative ? (object)snapshot.ExposureUs.Value : null,
                            ["exposure_epoch"] = snapshot.ExposureEpoch, ["coordinate_key"] = snapshot.Coordinate,
                            ["full_scale"] = snapshot.Quantitative ? (object)profile.fullScale : null,
                            ["pixel_pitch_um"] = snapshot.Quantitative && profile.pixelPitchUm > 0 ? (object)profile.pixelPitchUm : null,
                            ["timestamp_source"] = "host_transport_receive_only",
                            ["exposure_verified"] = snapshot.Quantitative, ["saturation_verified"] = snapshot.Quantitative,
                            ["metadata"] = snapshot.Metadata
                        }
                    };
                    Dictionary<string, object> changedCapabilities = null;
                    lock (reportLock) {
                        bool changed = !numericReported || !Equals(capabilities["preview_only"], !snapshot.Quantitative)
                            || !Equals(capabilities["ready"], snapshot.Quantitative && manualExposureOwned);
                        frameSamples.Enqueue(new { event_id = snapshot.EventId, snapshot_id = snapshot.SnapshotId,
                            marker = snapshot.Marker, width = snapshot.Width, height = snapshot.Height, source_status = snapshot.Status,
                            control_generation = snapshot.Generation,
                            minimum = range[0], maximum = range[1], callback_ms = snapshot.CallbackMs, copy_ms = snapshot.CopyMs,
                            frame_exposure_us = snapshot.ExposureUs, preview_only = !snapshot.Quantitative });
                        while (frameSamples.Count > 20) frameSamples.Dequeue();
                        lastCallbackMs = snapshot.CallbackMs; lastCopyMs = snapshot.CopyMs;
                        lastPublishMs = NativeRaster.Ms(Stopwatch.GetTimestamp() - started);
                        numericReported = true;
                        capabilities["numeric_images"] = true;
                        capabilities["preview_only"] = !snapshot.Quantitative;
                        capabilities["profile_verified"] = snapshot.Quantitative;
                        capabilities["exposure_verified"] = snapshot.Quantitative;
                        capabilities["saturation_verified"] = snapshot.Quantitative;
                        capabilities["frame_ids"] = snapshot.Quantitative;
                        capabilities["hardware_frame_ids"] = false;
                        capabilities["ready"] = snapshot.Quantitative && manualExposureOwned;
                        if (changed) changedCapabilities = new Dictionary<string, object>(capabilities);
                    }
                    // No SDK access in this worker. Metadata belongs to the completed callback.
                    if (snapshot.Generation != Interlocked.Read(ref controlGeneration) || paused) continue;
                    if (changedCapabilities != null) Emit(new { type = "capabilities", report = changedCapabilities });
                    Emit(new { type = "frame", descriptor });
                } catch (Exception failure) {
                    Emit(new { type = "error", message = failure.GetBaseException().Message, fatal = false });
                }
            }
        }

        private void Allocate(int bytes)
        {
            if (map != null) { view.Dispose(); map.Dispose(); }
            capacity = bytes;
            slotSpan = 8L + ((bytes + 7L) / 8) * 8;
            mapName = "Local\\BeamicPro_" + Guid.NewGuid().ToString("N");
            map = MemoryMappedFile.CreateNew(mapName, 3L * slotSpan);
            view = map.CreateViewAccessor();
            slot = -1;
        }

        public object Command(Dictionary<string, object> command)
        {
            string method = Convert.ToString(command["method"]);
            if (method == "set_rate") return CommandCore(command);
            lock (sdkLock) {
                bool wasPaused = paused;
                paused = true;
                long generation = Interlocked.Increment(ref controlGeneration);
                mailbox.Clear();
                Emit(new { type = "control_state", generation, phase = "transition", reason = method });
                gate.Close();
                bool resume = !wasPaused;
                try {
                    // Probe raster getters only while stopped, as required by official examples.
                    if (method == "probe" && resume) Invoke(PathValue("DataSource"), "Stop");
                    object result;
                    lock (reportLock) {
                        result = CommandCore(command);
                        if (ReferenceEquals(result, capabilities)) result = new Dictionary<string, object>(capabilities);
                    }
                    if (method == "probe" && resume) Invoke(PathValue("DataSource"), "Start");
                    if (method != "start" && method != "stop" && method != "shutdown") paused = wasPaused;
                    return result;
                } catch {
                    paused = true;
                    throw;
                } finally {
                    Interlocked.Exchange(ref nextAcceptTicks, 0);
                    Emit(new { type = "control_state", generation, phase = paused || stopped ? "paused" : "accepting", reason = method });
                    if (!paused && !stopped) gate.Open();
                }
            }
        }

        private object CommandCore(Dictionary<string, object> command)
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
                    bool quantitative = numericReported && !(bool)capabilities["preview_only"] && Verified;
                    capabilities["frame_ids"] = quantitative && profile.frameIdentityVerified;
                    capabilities["saturation_verified"] = quantitative && profile.saturationVerified;
                    capabilities["exposure_verified"] = quantitative && profile.frameExposureVerified;
                    capabilities["ready"] = quantitative && exposureControl && manualExposureOwned;
                    capabilities["members"] = DiscoverMembers();
                    capabilities["numeric_images"] = numericReported;
                    lock (reportLock) { capabilities["diagnostics"] = Diagnostics(); }
                    try {
                        settingExposureUs = Number(Egb("Get")) * unitToUs;
                        settingReadUtc = DateTime.UtcNow.ToString("o");
                        capabilities["current_exposure_setting_us"] = settingExposureUs;
                        capabilities["exposure_setting_note"] = "Current setting only; not frame-associated exposure";
                    } catch { capabilities["current_exposure_setting_us"] = null; }
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
                    settingExposureUs = Number(Egb("Get")) * unitToUs;
                    settingReadUtc = DateTime.UtcNow.ToString("o");
                    return new { exposure_us = Number(Egb("Get")) * unitToUs, epoch };
                }
            }
            throw new InvalidOperationException("Unsupported method: " + method);
        }

        public void Dispose()
        {
            paused = true;
            gate.Close(); // Timeout throws before touching resources that a callback may still use.
            stopped = true;
            mailbox.Clear();
            newFrame.Set();
            if (reader != null && !reader.Join(3000)) throw new TimeoutException("Snapshot publisher did not stop; bridge isolated");
            if (frameEvents != null && frameDelegate != null) {
                frameEvents.GetType().GetEvent("OnNewFrame")?.RemoveEventHandler(frameEvents, frameDelegate);
            }
            lock (sdkLock)
            {
                if (app != null) {
                    try { Invoke(PathValue("DataSource"), "Stop"); } catch { }
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

