using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using GD.MinMaxSlider;

/// <summary>
/// Represents and holds the agent's genes.
///
/// All genes are stored as values in [0, 1] and are converted to their actual value on usage.
/// A gene's actual value is defined by the range of that gene (set by the min max sliders). 
/// </summary>
public class AgentGenes : MonoBehaviour
{
    // Ranges
    [MinMaxSlider(1f, 4f)] public Vector2 maxFoodEnergyRange;

    public string genomeId;
    
    private float maxFoodEnergy = 1f;

    private float ScaleGeneFactorToRange(float geneFactor, Vector2 range)
    {
        return geneFactor * (range[1] - range[0]) + range[0];
    }

    public void Awake()
    {
        InitialiseIndirectGenes();
    }

    public void Inherit(AgentGenes fatherGenes, AgentGenes motherGenes)
    {
    }

    private void InitialiseIndirectGenes()
    {
    }

    public string GenomeId
    {
        get => genomeId;
        set => genomeId = value;
    }

    public float MaxFoodEnergy => maxFoodEnergy;
}