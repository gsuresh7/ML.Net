using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms;
using System;
using System.IO;
using System.Threading.Tasks;


namespace Pricing_App
{
    public class ModelHelper
    {
        /// <summary>
        /// Train and save model for predicting Pricing
        /// </summary>
        /// <param name="dataPath">Input training file path</param>
        /// <param name="outputModelPath">Trained model path</param>
        public static async Task TrainAndSaveModel(string dataPath, string outputModelPath = "Pricing_fastTree_model.zip")
        {
            if (File.Exists(outputModelPath))
            {
                File.Delete(outputModelPath);
            }

            var model = CreateProductModelUsingPipeline(dataPath);

            await model.WriteAsync(outputModelPath);
        }


        /// <summary>
        /// Build model for predicting pricing using Learning Pipelines API
        /// </summary>
        /// <param name="dataPath">Input training file path</param>
        /// <returns></returns>
        private static PredictionModel<PriceData, PriceUnitPrediction>
                        CreateProductModelUsingPipeline(string dataPath)
        {
            Console.WriteLine("*************************************************");
            Console.WriteLine("Training product forecasting model using Pipeline");

            //STEP 1:The learning pipeline loads all of the data and algorithms necessary to train the model. Add the following code into the Train method:
            var pipeline = new LearningPipeline();

            //STEP 2:The first step to perform is to load data from the training data set. In our case, 
            //training data set is stored in the csv file with a path defined by the _datapath field.
            //That file has the header with the column names, so the first row should be ignored while loading data.Columns in the file are separated by the comma(",").
            //Add the following code into the Train method:
            
            pipeline.Add(new TextLoader(dataPath).CreateFrom<PriceData>(useHeader: true, separator: ','));

            //STEP 3(Transformation):When the model is trained and evaluated, by default, the values in the Label column are considered as correct values to be predicted. 
            //As we want to predict the model price, copy the Model_Price column into the Label column. To do that, use ColumnCopier and add the following code:
            pipeline.Add(new ColumnCopier(("Model_Price", "Label")));

            //STEP 4(Transformation):The algorithm that trains the model requires numeric features, so you have to transform the categorical data(property_code, Avdrags_code, and payment_type) values into numbers.
            //To do that, use CategoricalOneHotVectorizer, which assigns different numeric key values to the different values in each of the columns, and add the following code:
            pipeline.Add(new CategoricalOneHotVectorizer("property_code",
                                             "Avdrags_code",
                                             "payment_type"));

            //STEP 5(Transformation):The last step in data preparation combines all of the feature columns into the Features column using the ColumnConcatenator transformation class. 
            //By default, a learning algorithm processes only features from the Features column.Add the following code:
            pipeline.Add(new ColumnConcatenator("Features",
                                     "property_code",
                                     "Avdrags_code",
                                     "property_count",
                                     "Loan_value",
                                     "payment_type"));

            //STEP 6(Choose right learner):After adding the data to the pipeline and transforming it into the correct input format, you select a learning algorithm(learner).
            //The learner trains the model. You chose a regression task for this problem, so you use a FastTreeRegressor learner, which is one of the regression learners provided by ML.NET.
            pipeline.Add(new FastTreeRegressor());

            // Convert the Label back into original text (after converting to number in step 3)
            //pipeline.Add(new PredictedLabelColumnOriginalValueConverter() { PredictedLabelColumn = "PredictedLabel" });

            // STEP 7(train Model): Train your model based on the data set
            var model = pipeline.Train<PriceData, PriceUnitPrediction>();

            return model;
        }

        /// <summary>
        /// Predict samples using saved model
        /// </summary>
        /// <param name="outputModelPath">Model file path</param>
        /// <returns></returns>
        public static async Task TestPrediction(string outputModelPath = "Pricing_fastTree_model.zip")
        {
            Console.WriteLine("*********************************");
            Console.WriteLine("Testing pricing Unit Forecast model");

            // Read the model that has been previously saved by the method SaveModel
            var model = await PredictionModel.ReadAsync<PriceData, PriceUnitPrediction>(outputModelPath);

            Console.WriteLine("** Testing Pricing 1 **");

            // STEP 8: Use your model to make a prediction
            // You can change these numbers to test different predictions
            PriceData dataSample = new PriceData()
            {
                property_code = "building",
                Avdrags_code = "1",
                property_count = 2,
                floor_Price = 1.3f,
                payment_type = "CRD",
                Model_Price = 0 // predict it. actual = 7
            };

            //model.Predict() predicts the indicative price to the one provided above
            PriceUnitPrediction prediction = model.Predict(dataSample);
            Console.WriteLine("indicative Pricing: {0}, actual Pricing: 7", prediction.Model_Price);

            
        }
    }
}
