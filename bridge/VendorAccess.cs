using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Reflection;

namespace BeamicPro.Bridge
{
    // This adapter is also exercised against a fake BeamMic-shaped API, without vendor DLLs.
    internal static class VendorAccess
    {
        public static object Value(object root, string path)
        {
            if (String.IsNullOrWhiteSpace(path)) throw new InvalidOperationException("No member path configured");
            object value = root;
            foreach (var name in path.Split('.')) {
                if (value == null) throw new InvalidOperationException("Null at " + path);
                var type = value.GetType();
                var property = type.GetProperty(name) ?? type.GetInterfaces().Select(i => i.GetProperty(name)).FirstOrDefault(p => p != null);
                if (property == null) throw new MissingMemberException(type.FullName, name);
                value = property.GetValue(value, null);
            }
            return value;
        }

        public static string Token(object value)
        {
            if (value == null) throw new InvalidOperationException("Null frame marker");
            var type = value.GetType();
            if (!(value is Guid || value is string || value is byte || value is sbyte || value is short || value is ushort
                  || value is int || value is uint || value is long || value is ulong))
                throw new InvalidOperationException("Unsupported frame marker type: " + type.FullName);
            var text = Convert.ToString(value, CultureInfo.InvariantCulture);
            if (String.IsNullOrWhiteSpace(text)) throw new InvalidOperationException("Empty frame marker");
            return type.FullName + ":" + text;
        }

        public static Dictionary<string, object> Describe(object root, IEnumerable<string> paths)
        {
            var result = new Dictionary<string, object>();
            foreach (var path in paths.Where(p => !String.IsNullOrWhiteSpace(p)).Distinct()) {
                try {
                    var value = Value(root, path);
                    object summary;
                    if (value is Array array) {
                        // Never serialize raster data into diagnostic reports.
                        summary = new { rank = array.Rank, length = array.Length,
                            dimensions = Enumerable.Range(0, array.Rank).Select(array.GetLength).ToArray(),
                            element_type = array.GetType().GetElementType().FullName };
                    } else if (value == null || value is string || value is bool || value is Guid || value.GetType().IsEnum
                               || value.GetType().IsPrimitive || value is decimal) {
                        summary = value == null ? null : Convert.ToString(value, CultureInfo.InvariantCulture);
                    } else { summary = "object; properties/methods listed separately"; }
                    result[path] = new { available = true, type = value?.GetType().FullName, value = summary };
                } catch (Exception error) {
                    result[path] = new { available = false, error = error.GetBaseException().Message };
                }
            }
            return result;
        }

        public static Dictionary<string, string[]> Methods(object root, IEnumerable<string> paths)
        {
            var result = new Dictionary<string, string[]>();
            foreach (var path in paths.Distinct()) {
                try {
                    var type = Value(root, path).GetType();
                    result[path] = type.GetMethods().Concat(type.GetInterfaces().SelectMany(i => i.GetMethods()))
                        .Where(m => !m.IsSpecialName && m.DeclaringType != typeof(object))
                        .Select(m => m.ReturnType.FullName + " " + m.Name + "(" + String.Join(", ",
                            m.GetParameters().Select(p => p.ParameterType.FullName + " " + p.Name)) + ")")
                        .Distinct().OrderBy(s => s).ToArray();
                } catch { result[path] = new string[0]; }
            }
            return result;
        }

        public static float[] Raster(Array array, int width, int height)
        {
            if (width <= 0 || height <= 0 || (long)width * height > 50000000)
                throw new InvalidOperationException("Invalid numeric image dimensions");
            if (array == null || array.Length != (long)width * height || array.Rank < 1 || array.Rank > 2
                || (array.Rank == 2 && (array.GetLength(0) != height || array.GetLength(1) != width)))
                throw new InvalidOperationException("Image array shape does not match Height x Width; orientation must be verified");
            var type = array.GetType().GetElementType();
            if (!(type == typeof(double) || type == typeof(float) || type == typeof(byte) || type == typeof(sbyte)
                  || type == typeof(short) || type == typeof(ushort) || type == typeof(int) || type == typeof(uint)
                  || type == typeof(long) || type == typeof(ulong)))
                throw new InvalidOperationException("Image member is not a numeric raster");
            var values = new float[array.Length];
            // Optimized common SDK cases, with finite checks after float conversion.
            if (array is double[] doubles) {
                for (int i = 0; i < values.Length; i++) values[i] = (float)doubles[i];
            } else if (array is float[] floats) {
                Array.Copy(floats, values, values.Length);
            } else {
                int i = 0;
                foreach (var item in array) values[i++] = Convert.ToSingle(item, CultureInfo.InvariantCulture);
            }
            if (values.Any(v => Single.IsNaN(v) || Single.IsInfinity(v)))
                throw new InvalidOperationException("Nonfinite or overflowing pixels");
            return values;
        }
    }
}
