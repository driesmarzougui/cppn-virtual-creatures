using System.Collections;
using System.Collections.Generic;
using UnityEngine;

/// <summary>
/// Updates the position of the sun to create a day / night cycle like behavior.
/// </summary>
public class SunController : MonoBehaviour
{
    public float speed = 10.0f;

    void Start()
    {
        // Set initial position -> Far in east and pointing towards west
        transform.position = new Vector3(1000, 0, 0);
        transform.rotation = Quaternion.Euler(0.0f, 270.0f, 0.0f);
    }

    void FixedUpdate()
    {
        // Change position based on time -> move counterclockwise around world
        transform.RotateAround(Vector3.zero, Vector3.forward, speed * Time.deltaTime);
    }
}