using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEditor;

[CustomEditor(typeof(SideChannelEventHandler))]
public class AgentCreatorEditor : Editor
{
    private int counter = 0;
    private void OnSceneGUI()
    {
        SideChannelEventHandler sceh = target as SideChannelEventHandler;

        if (Handles.Button(sceh.transform.position + Vector3.up * 5, Quaternion.LookRotation(Vector3.up), 1, 1,
            Handles.CylinderHandleCap))
        {
            sceh.AddGenome(counter.ToString());
            counter++;
        }
    }
}