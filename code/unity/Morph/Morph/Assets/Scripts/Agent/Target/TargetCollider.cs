using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class TargetCollider : MonoBehaviour
{
    private void OnTriggerEnter(Collider other)
    {
        if (other.attachedRigidbody && other.gameObject.CompareTag("agent"))
        {
            TargetController targetController = other.gameObject.GetComponentInParent<TargetController>();
            targetController.GoalReached();
            GetComponent<Collider>().enabled = false;
            Destroy(gameObject);
        }
    }
}
