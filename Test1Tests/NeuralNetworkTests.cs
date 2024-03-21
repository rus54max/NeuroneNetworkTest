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
            var outputs = new double[] { 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1 };
            var inputs = new double[,]
            {
                // Результат - Пациент болен - 1
                //             Пациент Здоров - 0

                // Неправильная температура T
                // Хороший возраст A
                // Курит S
                // Правильно питается F
                //T  A  S  F
                { 0, 0, 0, 0 },
                { 0, 0, 0, 1 },
                { 0, 0, 1, 0 },
                { 0, 0, 1, 1 },
                { 0, 1, 0, 0 },
                { 0, 1, 0, 1 },
                { 0, 1, 1, 0 },
                { 0, 1, 1, 1 },
                { 1, 0, 0, 0 },
                { 1, 0, 0, 1 },
                { 1, 0, 1, 0 },
                { 1, 0, 1, 1 },
                { 1, 1, 0, 0 },
                { 1, 1, 0, 1 },
                { 1, 1, 1, 0 },
                { 1, 1, 1, 1 }
            };

            Topology topology = new Topology(4, 1, 0.1, 2);
            NeuralNetwork neuronNetwork = new NeuralNetwork(topology);
            //neuronNetwork.Layers[1].Neurons[0].SetWeights(0.5, -0.1, 0.3, -0.1);
            double difference = neuronNetwork.Learn(outputs, inputs, 10000);

            List<double> result = new List<double>();
            for (int i = 0; i < outputs.Length; i++)
            {
                double[] row = NeuralNetwork.GetRow<double>(inputs, i);
                double res = neuronNetwork.FeedForward(row).Output;
                result.Add(res);
            }
            for (int x = 0; x < result.Count; x++)
            {
                double expected = Math.Round(outputs[x], 3);
                double actual = Math.Round(result[x], 3);
                Assert.AreEqual(expected, actual);
            }
        }

        [TestMethod()]
        public void DataSetTest()
        {
            List<double> outputs = new List<double>();
            List<double[]> inputs = new List<double[]>();
            using (StreamReader sr = new StreamReader("c:\\_Progects\\_ProgectNeuronNetwork\\Test1Tests\\heart.csv"))
            {
                var header = sr.ReadLine();
                while(!sr.EndOfStream)
                {
                    string row = sr.ReadLine();
                    List<double> value = row.Split(',').Select(v => Convert.ToDouble(v.Replace('.', ','))).ToList();
                    double output = value.Last();
                    double[] input = value.Take(value.Count - 1).ToArray();

                    outputs.Add(output);
                    inputs.Add(input);
                }
            }

            double[,] inputSignals = new double[inputs.Count, inputs[0].Length];
            for(int i = 0; i < inputSignals.GetLength(0); i++)
            {
                for (int j = 0; j < inputSignals.GetLength(1); j++)
                    inputSignals[i, j] = inputs[i][j];
            }

            Topology topology = new Topology(outputs.Count, 1, 0.1, outputs.Count / 2);
            NeuralNetwork neuralNetwork = new NeuralNetwork(topology);
            double difference = neuralNetwork.Learn(outputs.ToArray(), inputSignals, 1000);

            List<double> result = new List<double>();
            for (int i = 0; i < outputs.Count; i++)
            {
                double[] row = inputs[i];
                double res = neuralNetwork.FeedForward(row).Output;
                result.Add(res);
            }
            for (int x = 0; x < result.Count; x++)
            {
                double expected = Math.Round(outputs[x], 3);
                double actual = Math.Round(result[x], 3);
                Assert.AreEqual(expected, actual);
            }
        }
    }
}