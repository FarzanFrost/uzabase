from datasets import load_dataset
from pyspark.sql import SparkSession, Row
from pyspark import SparkConf, SparkContext 
from pyspark.rdd import RDD
from pyspark.sql.types import StructType, StructField, StringType, IntegerType
from pyspark.sql.functions import explode,split,col
from pyspark.sql.dataframe import DataFrame
from typing import List, Dict, Any
from datetime import datetime
import yaml
import os
import argparse
import random


class CountWords:
    # Load configuration from yaml.
    _config: Dict[Any, Any] = None

    @classmethod
    def load_config(cls, config_path: str) -> None:
        """
        Loads the configurations from cfg.yaml file to cls._config variable.

        Args:
            config_path (str): file path of cfg.yaml
        
        Returns:
            None
        """
        with open(config_path, "r") as file:
            cls._config = yaml.safe_load(file)

    # Based on a skim and scan, decided these preprocessing rules.
    @staticmethod
    def preprocess_data(text: str) -> str:
        """
        Does Preprocessing for each text line pass from the news dataset file.
        Removes trialing fullstops, commas, dashes, and adds a new line at the end of the text.

        Args:
            text (str): text line from dataset.
        
        Returns:
           text (str): preprocessed text.
        """
        # Remove trailing fullstops.
        text = text[:-1]

        # Remove commas.
        text = text.replace(',', '')

        # Remove Dashes.
        text = text.replace('-', '')

        # Add newline and return.
        return text + "\n"
    
    @classmethod
    def load_data(cls,) -> None:
        """
        Saves the test AS News dataset as a textfile after preprocessing.

        Args:
            None
        
        Returns:
            None
        """
        data_file_path: str = cls._config['data_file_path']

        # Load the AG News dataset, and extract only the test data.
        test_data = load_dataset(cls._config['hf_data_path'])["test"]

        # Write test data to text file
        with open(data_file_path, "w", encoding="utf-8") as f:
            for item in test_data:
                # Save only the "description" field
                f.write(
                    CountWords.preprocess_data(
                        item["description"]
                    )
                )

        print(f"Test data saved to {data_file_path}")
    
    @staticmethod
    def tokenize(line: str) -> List[str]:
        """
        splite text lines, into list of words.

        Args:
            line (str): dataset text line.

        Returns:
            words (List[str]): list of words.
        """
        return line.split(' ')
    
    # Count words RDD implementation.
    @classmethod
    def word_count_rdd(cls, data_file_path: str, output_file_name: str, count_all: bool) -> None:
        """
        Counts the numbers of words of given list of words, or all unique words using RDD.
        Saves the results as a parquet file.

        Args:
            data_file_path (str): file path of the text file that has the data.
            output_file_name (str): output parquet file name.
            count_all (bool): True to count all unique words, False to count words given as a list.
        
        Returns:
            None
        """
        print('Count words in RDD')
        # Configure SparkConf object.
        conf:SparkConf = SparkConf().setAppName(
            cls._config['spark_application_name']
        ).setMaster(cls._config['spark_master'])

        # Create SparkContext
        with SparkContext(conf=conf) as sc:
            log4jLogger = sc._jvm.org.apache.log4j
            sc.setLogLevel("INFO")
            
            # Create FileAppender
            log_file_name = 'Data_prcessed_all.txt' if count_all else 'Data_prcessed.txt'
            log_file_location = f'../logs/{log_file_name}'
            file_appender = log4jLogger.FileAppender()
            file_appender.setFile(log_file_location)  # Specify log file
            file_appender.setLayout(log4jLogger.PatternLayout("[%d] %p %c{1}: %m%n"))
            file_appender.activateOptions()

            LOGGER = log4jLogger.LogManager.getLogger(__name__)
            LOGGER.addAppender(file_appender)
            LOGGER.info("Spark Context Initialized")

            spark: SparkSession = SparkSession(sc)
            data_rdd: RDD[str] = sc.textFile(data_file_path) # Read data as RDD.

            # Convert RDD of lines to RDD of words
            word_rdd: RDD[str] = data_rdd.flatMap(CountWords.tokenize)

            if not count_all: # Filter only given unique words.
                unique_filter_words: List[str] = cls._config['unique_filter_words']
                word_rdd = word_rdd.filter(lambda word: word in unique_filter_words)

            single_word_count: int = 1

            word_counts_rdd = word_rdd.map(
                lambda word: (word, single_word_count) # Map word -> (word, 1)
            ).reduceByKey(
                lambda count1, count2 : count1 + count2 # reduce by key : (word, 1), (word, 1) -> (word, 2)
            )

            # Define schema to create DF.
            schema = StructType([
                StructField("word", StringType(), False),
                StructField("count", IntegerType(), False)
            ])

            # Create DF for results and save results.
            data_df = spark.createDataFrame(
                word_counts_rdd,
                schema=schema,
            )

            data_df.write.mode(cls._config['spark_mode']).parquet(output_file_name)

            # Show partial results or full results.
            data_df.show()

    # Count words DF implementation.
    @classmethod
    def word_count_df(cls, data_file_path: str, output_file_name: str, count_all: bool) -> None: 
        """
        Counts the numbers of words of given list of words, or all unique words using DF.
        Saves the results as a parquet file.

        Args:
            data_file_path (str): file path of the text file that has the data.
            output_file_name (str): output parquet file name.
            count_all (bool): True to count all unique words, False to count words given as a list.
        
        Returns:
            None
        """   
        print('Count words in DF')
        # Configure SparkConf object.
        conf:SparkConf = SparkConf().setAppName(
            cls._config['spark_application_name']
        ).setMaster(cls._config['spark_master'])

        # Create SparkContext
        with SparkContext(conf=conf) as sc:
            log4jLogger = sc._jvm.org.apache.log4j
            sc.setLogLevel("INFO")
            
            # Create FileAppender
            log_file_name = 'Data_prcessed_all.txt' if count_all else 'Data_prcessed.txt'
            log_file_location = f'../logs/{log_file_name}'
            file_appender = log4jLogger.FileAppender()
            file_appender.setFile(log_file_location)  # Specify log file
            file_appender.setLayout(log4jLogger.PatternLayout("[%d] %p %c{1}: %m%n"))
            file_appender.activateOptions()

            LOGGER = log4jLogger.LogManager.getLogger(__name__)
            LOGGER.addAppender(file_appender)
            LOGGER.info("Spark Context Initialized")

            spark: SparkSession = SparkSession(sc)

            # Read data as DF, where each line is a records of column named 'value'
            data_df: DataFrame = spark.read.text(data_file_path) 

            data_df = data_df.withColumn(
                'word', explode(split(col('value'), ' ')) # Split text line by space, and save words as records.
            )
            
            if not count_all: # Filter only given unique words.
                unique_filter_words: List[str] = cls._config['unique_filter_words']
                data_df = data_df.filter(col('word').isin(unique_filter_words))

            data_df = data_df.groupBy('word').count()
            
            data_df.write.mode(cls._config['spark_mode']).parquet(output_file_name)

            # Show partial results or full results.
            data_df.show()
    
    @classmethod
    def word_count(cls, output_file_name: str, count_all: bool=False) -> None:
        """
        Checks if data text file exists, if not loads the dataset into file.
        And randomly chooses if the word count to be done using RDD, or DF.

        Args:
            output_file_name (str): output parquet file name.
            count_all (bool): True to count all unique words, False to count words given as a list.
        
        Returns:
            None
        """
        # Load data file if not exists.
        if not os.path.exists(cls._config['data_file_path']):
            CountWords.load_data()
        
        data_file_path: str = cls._config['data_file_path']

        # Randomly choose one of the two methods to count words.
        # df -> using a dataframe.
        # rdd -> using a rdd.
        count_words_methods: List[str] = ['df', 'rdd']

        if random.choice(count_words_methods) == 'df':
            CountWords.word_count_df(
                data_file_path=data_file_path,
                count_all=count_all,
                output_file_name=output_file_name,
            )
        else:
            CountWords.word_count_rdd(
                data_file_path=data_file_path,
                count_all=count_all,
                output_file_name=output_file_name,
            )


if __name__ == "__main__":
    # Initialize and setup arg parser
    parser = argparse.ArgumentParser(description="News Data Processing")    
    parser.add_argument("command", choices=["process_data", "process_data_all"], help="Choose command to run")
    parser.add_argument("-cfg", "--config", required=True, help="Path to config file (YAML)")
    parser.add_argument("-dataset", "--dataset", required=True, help="Dataset name (should be 'news')")
    parser.add_argument("-dirout", "--output_dir", required=True, help="Output directory for saving results")

    # Get args.
    args = parser.parse_args()

    # Load config to CountWord class.
    config_path = args.config
    CountWords.load_config(config_path)

    output_file_directory: str = args.output_dir

    # Run appropriate function
    if args.command == "process_data":
        output_file_name: str = f'./{output_file_directory}/word_count_' + datetime.now().strftime("%Y%m%d") + '.parquet'
        CountWords.word_count(output_file_name,)
    elif args.command == "process_data_all":
        output_file_name: str = f'./{output_file_directory}/word_count_all_' + datetime.now().strftime("%Y%m%d") + '.parquet'
        CountWords.word_count(output_file_name, count_all=True)