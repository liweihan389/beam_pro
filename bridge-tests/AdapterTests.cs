using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Threading;
#if NETFRAMEWORK
using System.IO.MemoryMappedFiles;
using System.Web.Script.Serialization;
#endif

namespace BeamicPro.Bridge
{
    internal interface IFakeFrame { object UniqueKey { get; } }
    internal sealed class FakeFrame : IFakeFrame
    {
        public object Marker = Guid.NewGuid();
        public bool Changing = false;
        object IFakeFrame.UniqueKey => Changing ? Guid.NewGuid() : Marker;
        public int Width => 3;
        public int Height => 2;
        public int OriginalBpp => 12;
        public Array DoubleData { get; set; } = new double[] { 0, 1, 2, 3, 4, 5 };
        public bool IsBaselineCalibrated => false;
        public bool IsFrameCalibrated => false;
        public double BaseLineOffset => 0;
    }
    internal sealed class FakeInfo { public string Comment => "fake"; public string[] PartitionNames => new string[0]; }
    internal sealed class FakeSource {
        public string DataSource => "Fake camera";
        public string Status => "RUNNING";
        public Action DuringControl;
        public void Start() { DuringControl?.Invoke(); }
        public void Stop() { DuringControl?.Invoke(); }
    }
    internal sealed class FakeApp
    {
        public FakeFrame ResultsPriorityFrame { get; } = new FakeFrame();
        public FakeInfo FrameInfoResults { get; } = new FakeInfo();
        public FakeSource DataSource { get; } = new FakeSource();
    }

    internal static class AdapterTests
    {
        private static int assertions;
        private static void Check(bool value, string message) {
            assertions++;
            if (!value) throw new Exception(message);
        }
        private static void Reject(Action action, string message) {
            bool rejected = false;
            try { action(); } catch { rejected = true; }
            Check(rejected, message);
        }
        public static int Main()
        {
            try {
                var app = new FakeApp();
                Check((int)VendorAccess.Value(app, "ResultsPriorityFrame.Width") == 3, "Width is on the frame object");
                Check(VendorAccess.Value(app, "ResultsPriorityFrame.UniqueKey") is Guid, "Explicit interface properties must resolve");
                Reject(() => VendorAccess.Value(app, "FrameInfoResults.ID"), "Missing old ID must be reported");
                Check(VendorAccess.Token(ulong.MaxValue).Contains("18446744073709551615"), "Unsigned IDs must not overflow Int64");
                Check(VendorAccess.Token(Guid.NewGuid()).StartsWith("System.Guid:"), "Guid markers supported");
                Check(VendorAccess.Token("frame-a") != VendorAccess.Token("frame-b"), "String markers supported");
                Check(VendorAccess.Token(1) != VendorAccess.Token("1"), "Marker type is part of identity");
                Reject(() => VendorAccess.Token(null), "Null marker rejected");
                Reject(() => VendorAccess.Token(new object()), "Object identity is not a frame ID");
                Check(VendorAccess.Raster(new double[] { 0, 1, 2, 3, 4, 5 }, 3, 2).SequenceEqual(new float[] { 0, 1, 2, 3, 4, 5 }), "1D raster order");
                Check(VendorAccess.Raster(new ushort[,] { { 0, 1, 2 }, { 3, 4, 5 } }, 3, 2)[4] == 4, "2D Height x Width raster order");
                Reject(() => VendorAccess.Raster(new double[3, 2], 3, 2), "Do not silently transpose an unknown layout");
                Reject(() => VendorAccess.Raster(new double[] { Double.MaxValue }, 1, 1), "Float overflow rejected");
                Reject(() => VendorAccess.Raster(new double[] { Double.NaN }, 1, 1), "NaN rejected");
                Reject(() => VendorAccess.Raster(new string[] { "1" }, 1, 1), "String raster rejected");
                Reject(() => VendorAccess.Raster(new double[2], 3, 2), "Mismatched dimensions rejected");
                var fields = VendorAccess.Describe(app, new[] { "ResultsPriorityFrame.DoubleData", "FrameInfoResults.ID" });
                Check(fields.Count == 2, "Diagnostics retain successful and missing fields");
                var rasterDescription = fields["ResultsPriorityFrame.DoubleData"];
                var summary = rasterDescription.GetType().GetProperty("value").GetValue(rasterDescription, null);
                Check(summary.GetType().GetProperty("length").GetValue(summary, null).Equals(6), "Diagnostics describe array length without pixels");
                var source = new double[] { 1.00000000000001, 2, 3, 4, 5, 6 };
                var owned = NativeRaster.Own(source, 3, 2);
                source[0] = 999;
                Check(((double[])owned)[0] == 1.00000000000001, "Snapshot owns original precision before callback returns");
                var mailbox = new SnapshotMailbox();
                mailbox.Put(new NativeSnapshot { SnapshotId = 1 });
                mailbox.Put(new NativeSnapshot { SnapshotId = 2 });
                Check(mailbox.Replaced == 1 && mailbox.Take().SnapshotId == 2 && mailbox.Take() == null, "Latest snapshot mailbox is bounded");
                var gate = new SnapshotGate();
                Check(!gate.TryEnter(), "Closed gate never enters");
                gate.Open();
                Check(gate.TryEnter() && !gate.TryEnter(), "Reentrant callback skipped instead of waiting");
                Reject(() => gate.Close(5), "Control timeout never disposes a live callback");
                gate.Leave(); gate.Close(); gate.Open();
                Check(gate.TryEnter(), "Gate resumes after control"); gate.Leave();
#if NETFRAMEWORK
                ExerciseBridge(app);
#endif
                Console.WriteLine("Bridge adapter checks passed: " + assertions);
                return 0;
            } catch (Exception error) { Console.Error.WriteLine(error); return 1; }
        }

#if NETFRAMEWORK
        private static readonly BindingFlags Private = BindingFlags.Instance | BindingFlags.NonPublic;
        private static void Set(Bridge bridge, string name, object value) => typeof(Bridge).GetField(name, Private).SetValue(bridge, value);
        private static object Call(Bridge bridge, string name) => typeof(Bridge).GetMethod(name, Private).Invoke(bridge, null);
        private static void Until(Func<bool> condition) {
            var deadline = DateTime.UtcNow.AddSeconds(5);
            while (!condition()) {
                if (DateTime.UtcNow > deadline) throw new Exception("Bridge test timed out");
                Thread.Sleep(10);
            }
        }
        private static void ExerciseBridge(FakeApp app)
        {
            var writer = new StringWriter();
            var sync = TextWriter.Synchronized(writer);
            var profile = new DeviceProfile();
            var bridge = new Bridge(profile, sync);
            Set(bridge, "app", app); Set(bridge, "verifiedVersion", "2.0.0.0");
            Set(bridge, "capabilities", new Dictionary<string, object> { ["preview_only"] = true });
            Set(bridge, "paused", false);
            var gate = (SnapshotGate)typeof(Bridge).GetField("gate", Private).GetValue(bridge);
            gate.Open();
            var thread = new Thread(() => Call(bridge, "ReadFrames")) { IsBackground = true };
            Set(bridge, "reader", thread);
            Func<string> output = () => { lock (sync) { return writer.ToString(); } };
            Func<int> count = () => output().Split('\n').Count(l => l.Contains("\"type\":\"frame\""));
            try {
                // Capture before the worker starts, then mutate the vendor array immediately.
                Call(bridge, "FrameReady");
                ((double[])app.ResultsPriorityFrame.DoubleData)[4] = 999;
                thread.Start();
                Until(() => count() == 1);
                var json = new JavaScriptSerializer();
                var message = output().Split('\n').Where(l => l.Contains("\"type\":\"frame\""))
                    .Select(l => json.Deserialize<Dictionary<string, object>>(l)).First();
                var descriptor = (Dictionary<string, object>)message["descriptor"];
                var frame = (Dictionary<string, object>)descriptor["frame"];
                Check((bool)descriptor["preview_only"], "Default profile remains preview only");
                Check((string)descriptor["dtype"] == "<f8", "Raw transport keeps original SDK precision");
                Check(frame["exposure_us"] == null && frame["full_scale"] == null, "Unknown exposure/scale remain null");
                using (var map = MemoryMappedFile.OpenExisting((string)descriptor["map_name"]))
                using (var view = map.CreateViewAccessor()) {
                    var offset = Convert.ToInt64(descriptor["offset"]);
                    Check(view.ReadInt64(offset) == Convert.ToInt64(descriptor["sequence"]), "Shared memory sequence matches");
                    Check(view.ReadDouble(offset + 8 + 8 * 4) == 4, "Callback owned pixels before vendor mutated them");
                }
                Thread.Sleep(60); Call(bridge, "FrameReady"); Until(() => count() == 2);
                Check(Convert.ToInt64(typeof(Bridge).GetField("duplicateFrames", Private).GetValue(bridge)) >= 1, "Constant key counted diagnostically, not rejected");
                profile.frameIdentityVerified = true; // Obsolete flag must not resurrect key-based deduplication.
                Thread.Sleep(60); Call(bridge, "FrameReady"); Until(() => count() == 3);
                app.ResultsPriorityFrame.Changing = true;
                Thread.Sleep(60); Call(bridge, "FrameReady"); Until(() => count() == 4);
                Check(count() == 4, "UniqueKey is not used as a snapshot lock");
                // Simulate synchronous vendor callback from Start/Stop while a command is running.
                app.DataSource.DuringControl = () => Call(bridge, "FrameReady");
                bridge.Command(new Dictionary<string, object> { ["method"] = "start" });
                Check(count() == 4, "Control callback returns without taking the command lock or publishing");
                Thread.Sleep(60); Call(bridge, "FrameReady"); Until(() => count() == 5);
                Check(output().Contains("\"type\":\"control_state\""), "Control boundary emitted for host invalidation");
                var diagnostic = json.Serialize(Call(bridge, "Diagnostics"));
                Check(diagnostic.Contains("accepted_snapshots") && !diagnostic.Contains("\"pixels\""), "Bounded pixel-free diagnostics");
            } finally { bridge.Dispose(); }
        }

#endif
    }
}
