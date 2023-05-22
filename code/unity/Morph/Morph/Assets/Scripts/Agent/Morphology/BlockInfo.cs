namespace Agent.Morphology
{
    [System.Serializable]
    public class BlockInfo
    {
        public int type;
        public int x;
        public int y;
        public int z;
        public int sbd;
        public JointInfo[] joints;
    }
}