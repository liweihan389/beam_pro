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
    internal sealed class FakeSource { public string DataSource => "Fake camera"; }
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
            Set(bridge, "app", app);
            Set(bridge, "verifiedVersion", "2.0.0.0");
            Set(bridge, "capabilities", new Dictionary<string, object> { ["preview_only"] = true });
            Set(bridge, "paused", false);
            var thread = new Thread(() => Call(bridge, "ReadFrames")) { IsBackground = true };
            Set(bridge, "reader", thread);
            Func<string> output = () => { lock (sync) { return writer.ToString(); } };
            try {
                thread.Start();
                Call(bridge, "FrameReady");
                Until(() => output().Contains("\"type\":\"frame\""));
                var json = new JavaScriptSerializer();
                var message = output().Split('\n').Where(l => l.Contains("\"type\":\"frame\""))
                    .Select(l => json.Deserialize<Dictionary<string, object>>(l)).First();
                var descriptor = (Dictionary<string, object>)message["descriptor"];
                var frame = (Dictionary<string, object>)descriptor["frame"];
                Check((bool)descriptor["preview_only"], "Default profile must deliver preview without fake exposure/scale");
                Check(frame["exposure_us"] == null && frame["full_scale"] == null, "Unknown metadata remains null");
                Check(!(bool)frame["exposure_verified"] && !(bool)frame["saturation_verified"], "Preview is never quantitative");
                using (var map = MemoryMappedFile.OpenExisting((string)descriptor["map_name"]))
                using (var view = map.CreateViewAccessor()) {
                    var offset = Convert.ToInt64(descriptor["offset"]);
                    Check(view.ReadInt64(offset) == Convert.ToInt64(descriptor["sequence"]), "Shared memory sequence matches");
                    Check(view.ReadSingle(offset + 8 + 4 * 4) == 4, "Actual numeric raster delivered through shared memory");
                }
                var sequence = Convert.ToInt64(typeof(Bridge).GetField("sequence", Private).GetValue(bridge));
                // A key's name alone does not prove it identifies a camera frame.
                Call(bridge, "FrameReady");
                Until(() => output().Split('\n').Count(l => l.Contains("\"type\":\"frame\"")) == 2);
                Check(Convert.ToInt64(typeof(Bridge).GetField("sequence", Private).GetValue(bridge)) > sequence, "Unverified constant keys must not freeze event-triggered preview");
                profile.frameIdentityVerified = true;
                sequence = Convert.ToInt64(typeof(Bridge).GetField("sequence", Private).GetValue(bridge));
                Call(bridge, "FrameReady");
                Until(() => Convert.ToInt64(typeof(Bridge).GetField("duplicateFrames", Private).GetValue(bridge)) >= 2);
                Check(Convert.ToInt64(typeof(Bridge).GetField("sequence", Private).GetValue(bridge)) == sequence, "Verified duplicate marker is not delivered twice");
                app.ResultsPriorityFrame.Marker = "next-frame";
                Call(bridge, "FrameReady");
                Until(() => output().Split('\n').Count(l => l.Contains("\"type\":\"frame\"")) == 3);
                app.ResultsPriorityFrame.Changing = true;
                Call(bridge, "FrameReady");
                Until(() => Convert.ToInt64(typeof(Bridge).GetField("inconsistentFrames", Private).GetValue(bridge)) > 0);
                Check(output().Split('\n').Count(l => l.Contains("\"type\":\"frame\"")) == 3, "Changed-during-copy snapshot rejected");
                app.ResultsPriorityFrame.Changing = false;
                app.ResultsPriorityFrame.Marker = null;
                Call(bridge, "FrameReady");
                Until(() => output().Split('\n').Count(l => l.Contains("\"type\":\"frame\"")) == 4);
                Check(output().Contains("\"vendor_frame_id\":null"), "Unknown markers permit only explicitly unverified preview");
                var diagnostic = json.Serialize(Call(bridge, "Diagnostics"));
                Check(diagnostic.Contains("System.Guid") && diagnostic.Contains("frame_samples"), "Typed runtime diagnostics exported");
                Check(!diagnostic.Contains("\"pixels\""), "No pixels in diagnostic export");
            } finally { bridge.Dispose(); }
        }
#endif
    }
}
