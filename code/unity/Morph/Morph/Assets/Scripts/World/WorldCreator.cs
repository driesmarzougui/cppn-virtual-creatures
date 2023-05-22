using System;
using System.Collections;
using System.Collections.Generic;
using Unity.MLAgents;
using UnityEngine;

/// <summary>
/// Creates the Unity 3D world given the world file.
/// The world file is .txt file that defines a 2D grid with the blocks to be created at each position.
/// </summary>
public class WorldCreator : MonoBehaviour
{
    public GameObject dummyAgent;

    public int width = 100;
    public int depth = 100;

    void Start()
    {
        PersistantWorldManager.Instance.depth = depth;
        PersistantWorldManager.Instance.width = width;
        PersistantWorldManager.Instance.worldCreator = this;
        CreateDummy();
    }

    public void CreateDummy()
    {
        // Hacky way to make it possible to start the python script with an env.reset() without timing out
        Instantiate(dummyAgent, -3 * Vector3.one, Quaternion.identity, transform);
    }
}