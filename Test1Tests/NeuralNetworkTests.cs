using Microsoft.VisualStudio.TestTools.UnitTesting;
using Test1;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Test1.Tests
{
    [TestClass()]
    public class NeuralNetworkTests
    {
        [TestMethod()]
        public void FeedForwardTest()
        {
            Topology topology = new Topology(4, 1, 2);
            NeuralNetwork neuronNetwork = new NeuralNetwork(topology);
            neuronNetwork.Layers[1].Neurons[0].SetWeights(0.5, -0.1, 0.3, -0.1);
            neuronNetwork.Layers[1].Neurons[1].SetWeights(0.1, -0.3, 0.7, -0.3);
            neuronNetwork.Layers[2].Neurons[0].SetWeights(1.2, 0.8);

            Neuron neuron = neuronNetwork.FeedForward(new List<double>() { 1, 0, 0, 0 });

        }
    }
}