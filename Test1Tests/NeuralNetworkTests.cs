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
            var dataset = new List<Tuple<double, double[]>>
            { 
                // Результат - Пациент болен - 1
                //             Пациент Здоров - 0

                // Неправильная температура T
                // Хороший возраст A
                // Курит S
                // Правильно питается F
                //T  A  S  F
                new Tuple<double, double[]>(0, new double[] { 0, 0, 0, 0 }),
                new Tuple<double, double[]>(0, new double[] { 0, 0, 0, 1 }),
                new Tuple<double, double[]>(1, new double[] { 0, 0, 1, 0 }),
                new Tuple<double, double[]>(0, new double[] { 0, 0, 1, 1 }),
                new Tuple<double, double[]>(0, new double[] { 0, 1, 0, 0 }),
                new Tuple<double, double[]>(0, new double[] { 0, 1, 0, 1 }),
                new Tuple<double, double[]>(1, new double[] { 0, 1, 1, 0 }),
                new Tuple<double, double[]>(0, new double[] { 0, 1, 1, 1 }),
                new Tuple<double, double[]>(1, new double[] { 1, 0, 0, 0 }),
                new Tuple<double, double[]>(1, new double[] { 1, 0, 0, 1 }),
                new Tuple<double, double[]>(1, new double[] { 1, 0, 1, 0 }),
                new Tuple<double, double[]>(1, new double[] { 1, 0, 1, 1 }),
                new Tuple<double, double[]>(1, new double[] { 1, 1, 0, 0 }),
                new Tuple<double, double[]>(0, new double[] { 1, 1, 0, 1 }),
                new Tuple<double, double[]>(1, new double[] { 1, 1, 1, 0 }),
                new Tuple<double, double[]>(1, new double[] { 1, 1, 1, 1 } )
            };
            Topology topology = new Topology(4, 1, 0.1, 2);
            NeuralNetwork neuronNetwork = new NeuralNetwork(topology);
            //neuronNetwork.Layers[1].Neurons[0].SetWeights(0.5, -0.1, 0.3, -0.1);
            double difference = neuronNetwork.Learn(dataset, 100000);

            List<double> result = new List<double>();
            foreach (var data in dataset)
                result.Add(neuronNetwork.FeedForward(data.Item2).Output);
            for (int x = 0; x < result.Count; x++)
            {
                double expected = Math.Round(dataset[x].Item1, 3);
                double actual = Math.Round(result[x], 3);
                Assert.AreEqual(expected, actual);
            }
        }
    }
}