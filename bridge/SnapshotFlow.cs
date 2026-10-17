using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Threading;

namespace BeamicPro.Bridge
{
    // No lock here is held while calling vendor code or waiting for a callback.
    internal sealed class SnapshotGate
    {
        private readonly object sync = new object();
        private readonly ManualResetEventSlim idle = new ManualResetEventSlim(true);
        private bool open;
        private bool active;
        public bool TryEnter() {
            lock (sync) {
                if (!open || active) return false;
                active = true; idle.Reset(); return true;
            }
        }
        public void Leave() { lock (sync) { active = false; idle.Set(); } }
        public void Close(int timeoutMs = 3000) {
            lock (sync) { open = false; }
            if (!idle.Wait(timeoutMs)) throw new TimeoutException("Callback did not finish; bridge must be isolated, not disposed under a live callback");
        }
        public void Open() { lock (sync) { open = true; } }
    }

    internal sealed class NativeSnapshot
    {
        public Array Pixels;
        public int Width, Height;
        public long EventId, SnapshotId, Generation;
        public double CopyMs, CallbackMs;
        public string Marker, Coordinate, Source, Status;
        public bool Quantitative;
        public double? ExposureUs;
        public long ExposureEpoch;
        public Dictionary<string, object> Metadata;
    }

    internal sealed class SnapshotMailbox
    {
        private NativeSnapshot pending;
        public long Replaced;
        public void Put(NativeSnapshot value) {
            if (Interlocked.Exchange(ref pending, value) != null) Interlocked.Increment(ref Replaced);
        }
        public NativeSnapshot Take() => Interlocked.Exchange(ref pending, null);
        public void Clear() { Interlocked.Exchange(ref pending, null); }
    }

    internal static class NativeRaster
    {
        public static Array Own(Array source, int width, int height) {
            if (source == null || width <= 0 || height <= 0 || (long)width * height > 50000000
                || source.Length != (long)width * height || source.Rank < 1 || source.Rank > 2
                || (source.Rank == 2 && (source.GetLength(0) != height || source.GetLength(1) != width)))
                throw new InvalidOperationException("Invalid Height x Width raster; layout needs verification");
            Dtype(source.GetType().GetElementType()); // Reject nonnumeric/object arrays before copying.
            return (Array)source.Clone();
        }
        public static string Dtype(Type type) {
            if (type == typeof(double)) return "<f8";
            if (type == typeof(float)) return "<f4";
            if (type == typeof(int)) return "<i4";
            if (type == typeof(uint)) return "<u4";
            if (type == typeof(short)) return "<i2";
            if (type == typeof(ushort)) return "<u2";
            if (type == typeof(byte)) return "|u1";
            if (type == typeof(sbyte)) return "|i1";
            if (type == typeof(long)) return "<i8";
            if (type == typeof(ulong)) return "<u8";
            throw new InvalidOperationException("Unsupported numeric element type: " + type.FullName);
        }
        public static double[] Range(Array array) {
            double minimum = Double.PositiveInfinity, maximum = Double.NegativeInfinity;
            if (array is double[] doubles) {
                foreach (double value in doubles) {
                    if (Double.IsNaN(value) || Double.IsInfinity(value)) throw new InvalidOperationException("Nonfinite pixels");
                    minimum = Math.Min(minimum, value); maximum = Math.Max(maximum, value);
                }
            } else {
                foreach (var item in array) {
                    double value = Convert.ToDouble(item);
                    if (Double.IsNaN(value) || Double.IsInfinity(value)) throw new InvalidOperationException("Nonfinite pixels");
                    minimum = Math.Min(minimum, value); maximum = Math.Max(maximum, value);
                }
            }
            return new[] { minimum, maximum };
        }
        public static double Ms(long ticks) => ticks * 1000.0 / Stopwatch.Frequency;
    }
}
