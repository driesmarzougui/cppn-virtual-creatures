using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class VegetationSpawner : MonoBehaviour
{
    public GameObject vegetationPrefab;

    private GameObject vegetation; 
    // Start is called before the first frame update
    private void Awake()
    {
        vegetation = Instantiate(vegetationPrefab, vegetationPrefab.transform.position, vegetationPrefab.transform.rotation, transform.parent);
    }
    private void OnDestroy()
    {
       Destroy(vegetation); 
    }

    public GameObject Vegetation => vegetation;
}
