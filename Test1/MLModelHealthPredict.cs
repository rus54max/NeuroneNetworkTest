using Microsoft.ML;
using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Test1
{
    public static class MLModelHealthPredict
    {
        public static void Predict()
        {
            var mlContext = new MLContext();
            var sampleData = new List<InputData>()
            {
                new InputData{ Features = new float[] { 0, 0, 0, 0 }, Diagnosis = "Healthy"},
                new InputData{ Features = new float[] { 0, 0, 0, 1 }, Diagnosis = "Healthy"},
                new InputData{ Features = new float[] { 0, 0, 1, 0 }, Diagnosis = "Diseased"},
                new InputData{ Features = new float[] { 0, 0, 1, 1 }, Diagnosis = "Healthy"},
                new InputData{ Features = new float[] { 0, 1, 0, 0 }, Diagnosis = "Healthy"},
                new InputData{ Features = new float[] { 0, 1, 0, 1 }, Diagnosis = "Healthy"},
                new InputData{ Features = new float[] { 0, 1, 1, 0 }, Diagnosis = "Diseased"},
                new InputData{ Features = new float[] { 0, 1, 1, 1 }, Diagnosis = "Healthy"},
                new InputData{ Features = new float[] { 1, 0, 0, 0 }, Diagnosis = "Diseased"},
                new InputData{ Features = new float[] { 1, 0, 0, 1 }, Diagnosis = "Diseased"},
                new InputData{ Features = new float[] { 1, 0, 1, 0 }, Diagnosis = "Diseased"},
                new InputData{ Features = new float[] { 1, 0, 1, 1 }, Diagnosis = "Diseased"},
                new InputData{ Features = new float[] { 1, 1, 0, 0 }, Diagnosis = "Diseased"},
                new InputData{ Features = new float[] { 1, 1, 0, 1 }, Diagnosis = "Healthy"},
                new InputData{ Features = new float[] { 1, 1, 1, 0 }, Diagnosis = "Diseased"},
                new InputData{ Features = new float[] { 1, 1, 1, 1 }, Diagnosis = "Diseased"}
            }.AsEnumerable();
            //InputData[] imMemoryCollection = new InputData[]
            //{
            //    new InputData{ Features = new float[] { 0, 0, 0, 0 }, Diagnosis = 0},
            //    new InputData{ Features = new float[] { 0, 0, 0, 1 }, Diagnosis = 0},
            //    new InputData{ Features = new float[] { 0, 0, 1, 0 }, Diagnosis = 1},
            //    new InputData{ Features = new float[] { 0, 0, 1, 1 }, Diagnosis = 0},
            //    new InputData{ Features = new float[] { 0, 1, 0, 0 }, Diagnosis = 0},
            //    new InputData{ Features = new float[] { 0, 1, 0, 1 }, Diagnosis = 0},
            //    new InputData{ Features = new float[] { 0, 1, 1, 0 }, Diagnosis = 1},
            //    new InputData{ Features = new float[] { 0, 1, 1, 1 }, Diagnosis = 0},
            //    new InputData{ Features = new float[] { 1, 0, 0, 0 }, Diagnosis = 1},
            //    new InputData{ Features = new float[] { 1, 0, 0, 1 }, Diagnosis = 1},
            //    new InputData{ Features = new float[] { 1, 0, 1, 0 }, Diagnosis = 1},
            //    new InputData{ Features = new float[] { 1, 0, 1, 1 }, Diagnosis = 1},
            //    new InputData{ Features = new float[] { 1, 1, 0, 0 }, Diagnosis = 1},
            //    new InputData{ Features = new float[] { 1, 1, 0, 1 }, Diagnosis = 0},
            //    new InputData{ Features = new float[] { 1, 1, 1, 0 }, Diagnosis = 1},
            //    new InputData{ Features = new float[] { 1, 1, 1, 1 }, Diagnosis = 1}
            //};
            var data = mlContext.Data.LoadFromEnumerable(sampleData);
            //var data = mlContext.Data.LoadFromEnumerable(imMemoryCollection);
            var pipeLine = mlContext.Transforms.Conversion.MapValueToKey("Label")
                .Append(mlContext.MulticlassClassification.Trainers.LbfgsMaximumEntropy())
                .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

            var testTrainSplit = mlContext.Data.TrainTestSplit(data, testFraction: 0.2);
            var model = pipeLine.Fit(testTrainSplit.TrainSet);

            var predictions = model.Transform(testTrainSplit.TrainSet);
            var metrics = mlContext.MulticlassClassification.Evaluate(predictions);

            Console.WriteLine($"Score = {metrics.MicroAccuracy}");

            InputData inputData = new InputData { Features = new float[] { 0, 0, 1, 1 } };
            var predictionFunc = mlContext.Model.CreatePredictionEngine<InputData, OutputData>(model);
            var prediction = predictionFunc.Predict(inputData);
                Console.WriteLine($"Patient is - {prediction.Prediction}");
        }
    }
    public class InputData
    {
        [ColumnName("Features"), VectorType(4)]
        public float[] Features;
        [ColumnName("Label")]
        public string Diagnosis;
    }
    public class OutputData
    {
        [ColumnName("PredictedLabel")]
        public string Prediction;
    }
}
