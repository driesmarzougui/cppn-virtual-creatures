using UnityEngine;

namespace Agent.Utils
{
    public static class RotationUtils
    {
        public static Vector3 GetJointLocalRotationEulerNormalised(ConfigurableJoint joint)
        {
            Vector3 rotEuler = joint.transform.localRotation.eulerAngles;
            // Negative rotations are >= 180f -> convert them back to negative representation (e.g. 340 -> -20 degrees)
            // Scale to [-1 ; 1] where -1.0 == max negative angle, 0.0 == zero angle and 1.0 == max positive angle

            float lax = joint.lowAngularXLimit.limit;
            float hax = joint.highAngularXLimit.limit;
            float ayl = joint.angularYLimit.limit;
            float azl = joint.angularZLimit.limit;

            rotEuler.x = rotEuler.x >= 180f
                ? (lax < -3f ? 
                    -1 * (rotEuler.x - 360f) / lax : 0f)
                : (hax > 3f ? rotEuler.x / hax : 0f);
            rotEuler.y = ayl > 3f ? (rotEuler.y >= 180f ? rotEuler.y - 360f : rotEuler.y) / ayl : 0f;
            rotEuler.z = azl > 3f ? (rotEuler.z >= 180f ? rotEuler.z - 360f : rotEuler.z) / azl : 0f;

            return rotEuler;
        }

        /// <summary>
        /// Sets a joint's targetRotation to match a given local rotation.
        /// </summary>
        public static void SetTargetRotationLocal(ConfigurableJoint joint, Quaternion targetLocalRotation)
        {
            SetTargetRotationInternal(joint, targetLocalRotation, joint.transform.localRotation, Space.Self);
        }

        /// <summary>
        /// Sets a joint's targetRotation to match a given world rotation.
        /// The joint transform's world rotation must be cached on Start and passed into this method.
        /// </summary>
        public static void SetTargetRotation(ConfigurableJoint joint, Quaternion targetWorldRotation,
            Quaternion startWorldRotation)
        {
            if (!joint.configuredInWorldSpace)
            {
                Debug.LogError(
                    "SetTargetRotation must be used with joints that are configured in world space. For local space joints, use SetTargetRotationLocal.",
                    joint);
            }

            SetTargetRotationInternal(joint, targetWorldRotation, startWorldRotation, Space.World);
        }

        static void SetTargetRotationInternal(ConfigurableJoint joint, Quaternion targetRotation,
            Quaternion startRotation, Space space)
        {
            // Calculate the rotation expressed by the joint's axis and secondary axis
            var right = joint.axis;
            var forward = Vector3.Cross(joint.axis, joint.secondaryAxis).normalized;
            var up = Vector3.Cross(forward, right).normalized;
            Quaternion worldToJointSpace = Quaternion.LookRotation(forward, up);

            // Transform into world space
            Quaternion resultRotation = Quaternion.Inverse(worldToJointSpace);

            // Counter-rotate and apply the new local rotation.
            // Joint space is the inverse of world space, so we need to invert our value
            if (space == Space.World)
            {
                resultRotation *= startRotation * Quaternion.Inverse(targetRotation);
            }
            else
            {
                resultRotation *= Quaternion.Inverse(targetRotation) * startRotation;
            }

            // Transform back into joint space
            resultRotation *= worldToJointSpace;

            // Set target rotation to our newly calculated rotation
            joint.targetRotation = resultRotation;
        }
    }
}