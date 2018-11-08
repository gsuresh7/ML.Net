using Microsoft.ML.Runtime.Api;

namespace Pricing_App
{
    public class PriceData
    {
        [Column("0")]
        public string property_code;

        [Column("1")]
        public string Avdrags_code;

        [Column("2")]
        public float property_count;

        [Column("3")]
        public float Loan_value;

        [Column("4")]
        public float floor_Price;

        [Column("5")]
        public string payment_type;

        [Column("6")]
        public float Model_Price;
    }

    public class PriceUnitPrediction
    {
        [ColumnName("Score")]
        public float Model_Price;
    }
}
