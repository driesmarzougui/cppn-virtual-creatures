using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Rotator : MonoBehaviour
{
    // Start is called before the first frame update

    private ConfigurableJoint joint;
    private float timer;
    private bool rotated;
    void Start()
    {
        joint = GetComponent<ConfigurableJoint>();
    }

    // Update is called once per frame
    void Update()
    {
	    Debug.Log("LOCAL POSITION: " + (joint.transform.position - transform.parent.position));

	    Vector3 rotEuler = joint.transform.localRotation.eulerAngles;
	    // Negative rotations are >= 180f -> convert them back to negative representation (e.g. 340 -> -20 degrees)
	    rotEuler.x = rotEuler.x >= 180f ? rotEuler.x - 360f : rotEuler.x;
	    rotEuler.y = rotEuler.y >= 180f ? rotEuler.y - 360f : rotEuler.y;
	    rotEuler.z = rotEuler.z >= 180f ? rotEuler.z - 360f: rotEuler.z;
	    
	    // Scale to [-1 ; 1] where -1.0 == max negative angle, 0.0 == zero angle and 1.0 == max positive angle
	    rotEuler /= 75f;
	    
		Debug.Log("ROTATION:       " + rotEuler); 
        timer += Time.deltaTime;
        if (timer > 2f)
        {
	        /*
	        Debug.Log("rotating...");
	        Quaternion rotation = Quaternion.AngleAxis(50, transform.forward).normalized;
	        Quaternion newTRotation = (rotation * joint.targetRotation).normalized;
	        joint.targetRotation = newTRotation;

	        
	        rotated = true;*/
	        
            if (Mathf.Sin(timer) > 0)
            {
	            Quaternion rotation = Quaternion.Euler(60f, 0f, 0f);
	            SetTargetRotationLocal(joint, rotation, joint.transform.localRotation);
            }
            else
            {
	            
	            Quaternion rotation = Quaternion.Euler(0f, 0f, 0f);
	            SetTargetRotationLocal(joint, rotation, joint.transform.localRotation);
            }
        }
    }
    
    
    	/// <summary>
	/// Sets a joint's targetRotation to match a given local rotation.
	/// The joint transform's local rotation must be cached on Start and passed into this method.
	/// </summary>
	public static void SetTargetRotationLocal (ConfigurableJoint joint, Quaternion targetLocalRotation, Quaternion startLocalRotation)
	{
		if (joint.configuredInWorldSpace) {
			Debug.LogError ("SetTargetRotationLocal should not be used with joints that are configured in world space. For world space joints, use SetTargetRotation.", joint);
		}
		SetTargetRotationInternal (joint, targetLocalRotation, startLocalRotation, Space.Self);
	}
	
	/// <summary>
	/// Sets a joint's targetRotation to match a given world rotation.
	/// The joint transform's world rotation must be cached on Start and passed into this method.
	/// </summary>
	public static void SetTargetRotation (ConfigurableJoint joint, Quaternion targetWorldRotation, Quaternion startWorldRotation)
	{
		if (!joint.configuredInWorldSpace) {
			Debug.LogError ("SetTargetRotation must be used with joints that are configured in world space. For local space joints, use SetTargetRotationLocal.", joint);
		}
		SetTargetRotationInternal (joint, targetWorldRotation, startWorldRotation, Space.World);
	}
	
	static void SetTargetRotationInternal (ConfigurableJoint joint, Quaternion targetRotation, Quaternion startRotation, Space space)
	{
		// Calculate the rotation expressed by the joint's axis and secondary axis
		var right = joint.axis;
		var forward = Vector3.Cross (joint.axis, joint.secondaryAxis).normalized;
		var up = Vector3.Cross (forward, right).normalized;
		Quaternion worldToJointSpace = Quaternion.LookRotation (forward, up);
		
		// Transform into world space
		Quaternion resultRotation = Quaternion.Inverse (worldToJointSpace);
		
		// Counter-rotate and apply the new local rotation.
		// Joint space is the inverse of world space, so we need to invert our value
		if (space == Space.World) {
			resultRotation *= startRotation * Quaternion.Inverse (targetRotation);
		} else {
			resultRotation *= Quaternion.Inverse (targetRotation) * startRotation;
		}
		
		// Transform back into joint space
		resultRotation *= worldToJointSpace;
		
		// Set target rotation to our newly calculated rotation
		joint.targetRotation = resultRotation;
	}
}