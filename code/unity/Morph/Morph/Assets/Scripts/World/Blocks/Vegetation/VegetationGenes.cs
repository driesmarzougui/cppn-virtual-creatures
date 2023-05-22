using System.Collections;
using System.Collections.Generic;
using UnityEngine;

/// <summary>
/// Represents and holds the vegetation's genes.
///
/// All genes are stored as values in [0, 1].
/// </summary>
public class VegetationGenes : MonoBehaviour
{
    
    public float maxEnergy;
    public float minSpreadEnergy;
    public int spreadRadius;
    public float probability;
    public float energyGainPerTimestep;
    public int foodType;
    
    public void Inherit(VegetationGenes parentGenes)
    {
        maxEnergy = parentGenes.maxEnergy;
        minSpreadEnergy = parentGenes.minSpreadEnergy;
        spreadRadius = parentGenes.spreadRadius;
        probability = parentGenes.probability;
        energyGainPerTimestep = parentGenes.energyGainPerTimestep;
        foodType = parentGenes.foodType;
    }
}
