namespace Agent.Morphology
{
    [System.Serializable]
    public class MorphInfo
    {
        public int genomeId;
        public float agentSpaceWidth;
        public float agentSpaceHeight;
        public float agentSpaceDepth;
        public float agentSubSpaceWidth;
        public float agentSubSpaceHeight;
        public float agentSubSpaceDepth;
        public BlockInfo[] blocks;
    }
}