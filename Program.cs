using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms;
using System.Collections.Generic;
using System;
using System.Threading.Tasks;
using System.Linq;

namespace Pricing_App
{
    class Program
    {

        static async Task Main(string[] args)
        {
            try
            {
                await ModelHelper.TrainAndSaveModel("data/Train_DataSet.csv");
                await ModelHelper.TestPrediction();
                Console.Write("Hit any key to exit");
                Console.ReadLine();

            }
            catch (Exception ex)
            {
                Console.Write(ex.Message);
            }
        }

     
    }
}