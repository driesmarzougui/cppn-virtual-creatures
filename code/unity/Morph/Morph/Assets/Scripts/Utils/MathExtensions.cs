using UnityEngine;

namespace Agent.Utils
{
    public static class MathExtensions
    {
        public static bool ValueInRange(int thisValue, int min, int max)
        {
            return min <= thisValue && thisValue <= max;
        }

        public static bool ValuesInRange(Vector3Int thisValue, int min, int max)
        {
            return min <= thisValue.x && thisValue.x <= max
                                       && min <= thisValue.y && thisValue.y <= max
                                       && min <= thisValue.z && thisValue.z <= max;
        }

        public static bool ValuesInRange(Vector3 thisValue, float min, float max)
        {
            return min <= thisValue.x && thisValue.x <= max
                                       && min <= thisValue.y && thisValue.y <= max
                                       && min <= thisValue.z && thisValue.z <= max;
        }

        public static Vector3 Vector3Division(Vector3 a, Vector3 b)
        {
            return new Vector3(a.x / b.x, a.y / b.y, a.z / b.z);
        }
    }
}