import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import os
from scipy import stats

class AircraftAnomalyDetector:
    def __init__(self, data_path, output_dir="anomaly_results"):
        """
        Initialize the anomaly detector with the path to the data file.
        
        Args:
            data_path (str): Path to the CSV file with semicolon delimiter
            output_dir (str): Directory to save results
        """
        self.data_path = data_path
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Load and preprocess data
        self.load_data()
        
    def load_data(self):
        """Load and preprocess the data."""
        # Load data with semicolon delimiter
        self.data = pd.read_csv(self.data_path, sep=';')
        
        # Convert timestamp to datetime
        self.data['timestamp'] = pd.to_datetime(self.data['timestamp'])
        
        # Sort by timestamp
        self.data = self.data.sort_values('timestamp')
        
        print(f"Data loaded successfully. Shape: {self.data.shape}")
        print(f"Systems found: {self.data['system'].unique()}")
        print(f"Date range: {self.data['timestamp'].min()} to {self.data['timestamp'].max()}")
        
    def explore_data(self):
        """Perform basic exploratory analysis of the data."""
        # Basic statistics for y_axis values grouped by system
        stats_by_system = self.data.groupby('system')['y_axis'].agg(['count', 'mean', 'std', 'min', 'max'])
        
        # Save statistics to CSV
        stats_by_system.to_csv(f"{self.output_dir}/system_statistics.csv")
        
        # Plot distribution of y_axis values by system
        plt.figure(figsize=(12, 8))
        for system in self.data['system'].unique():
            sns.kdeplot(self.data[self.data['system'] == system]['y_axis'], label=system)
        
        plt.title('Distribution of y_axis Values by System')
        plt.xlabel('y_axis Value')
        plt.ylabel('Density')
        plt.legend()
        plt.savefig(f"{self.output_dir}/y_axis_distribution_by_system.png")
        
        # Create boxplots for each system to visualize outliers
        plt.figure(figsize=(14, 8))
        sns.boxplot(x='system', y='y_axis', data=self.data)
        plt.title('y_axis Values by System (Boxplot)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/y_axis_boxplot_by_system.png")
        
        # Return the summary statistics
        return stats_by_system
    
    def detect_statistical_outliers(self, z_threshold=3.0):
        """
        Detect outliers using z-score method within each system and subsystem.
        
        Args:
            z_threshold (float): Z-score threshold for outlier detection
            
        Returns:
            DataFrame: Data with outlier flags
        """
        # Create a copy of the data
        data_with_outliers = self.data.copy()
        data_with_outliers['is_outlier'] = False
        data_with_outliers['z_score'] = np.nan
        
        # Calculate z-scores for each system-subsystem combination
        for (sys, subsys), group in self.data.groupby(['system', 'subsystem']):
            if len(group) >= 2:  # Need at least 2 points to calculate z-score
                z_scores = np.abs(stats.zscore(group['y_axis']))
                
                # Get indices from the original dataframe
                indices = group.index
                
                # Assign z-scores and outlier flags
                data_with_outliers.loc[indices, 'z_score'] = z_scores
                data_with_outliers.loc[indices, 'is_outlier'] = z_scores > z_threshold
        
        # Save outliers to CSV
        outliers = data_with_outliers[data_with_outliers['is_outlier']]
        outliers.to_csv(f"{self.output_dir}/statistical_outliers.csv", index=False)
        
        print(f"Statistical analysis complete. Found {len(outliers)} outliers.")
        
        return data_with_outliers
    
    def detect_anomalies_isolation_forest(self, contamination=0.05):
        """
        Detect anomalies using Isolation Forest algorithm for each system separately.
        
        Args:
            contamination (float): Expected proportion of anomalies
            
        Returns:
            DataFrame: Data with anomaly flags
        """
        # Create a copy of the data
        data_with_anomalies = self.data.copy()
        data_with_anomalies['is_anomaly_if'] = False
        
        # For each system, apply Isolation Forest
        for system, group in self.data.groupby('system'):
            if len(group) >= 10:  # Need enough data points for meaningful analysis
                # Extract features
                X = group[['y_axis']].values
                
                # Standardize features
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                
                # Apply Isolation Forest
                clf = IsolationForest(contamination=contamination, random_state=42)
                y_pred = clf.fit_predict(X_scaled)
                
                # Flag anomalies (-1 indicates anomaly)
                is_anomaly = y_pred == -1
                
                # Get indices from the original dataframe
                indices = group.index
                
                # Assign anomaly flags
                data_with_anomalies.loc[indices, 'is_anomaly_if'] = is_anomaly
        
        # Save anomalies to CSV
        anomalies = data_with_anomalies[data_with_anomalies['is_anomaly_if']]
        anomalies.to_csv(f"{self.output_dir}/isolation_forest_anomalies.csv", index=False)
        
        print(f"Isolation Forest analysis complete. Found {len(anomalies)} anomalies.")
        
        return data_with_anomalies
    
    def detect_time_series_anomalies(self, window_size=5, threshold=3.0):
        """
        Detect anomalies in time series data using moving average and standard deviation.
        
        Args:
            window_size (int): Size of the rolling window
            threshold (float): Number of standard deviations for anomaly threshold
            
        Returns:
            DataFrame: Data with time series anomaly flags
        """
        # Create a copy of the data
        data_with_ts_anomalies = self.data.copy()
        data_with_ts_anomalies['is_ts_anomaly'] = False
        data_with_ts_anomalies['ma'] = np.nan
        data_with_ts_anomalies['ma_std'] = np.nan
        
        # For each system-subsystem combination, analyze time series
        for (sys, subsys), group in self.data.groupby(['system', 'subsystem']):
            if len(group) >= window_size * 2:  # Need enough data points
                # Sort by timestamp
                ts_data = group.sort_values('timestamp')
                
                # Calculate rolling mean and std
                rolling = ts_data['y_axis'].rolling(window=window_size)
                ma = rolling.mean()
                ma_std = rolling.std()
                
                # Get indices from the original dataframe
                indices = ts_data.index
                
                # Assign moving average and std
                data_with_ts_anomalies.loc[indices, 'ma'] = ma
                data_with_ts_anomalies.loc[indices, 'ma_std'] = ma_std
                
                # Flag as anomaly if value is outside threshold * std from the mean
                # Skip the first window_size-1 points as they don't have enough history
                valid_indices = ts_data.index[window_size-1:]
                for idx in valid_indices:
                    value = data_with_ts_anomalies.loc[idx, 'y_axis']
                    mean = data_with_ts_anomalies.loc[idx, 'ma']
                    std = data_with_ts_anomalies.loc[idx, 'ma_std']
                    
                    if std > 0 and abs(value - mean) > threshold * std:
                        data_with_ts_anomalies.loc[idx, 'is_ts_anomaly'] = True
        
        # Save time series anomalies to CSV
        ts_anomalies = data_with_ts_anomalies[data_with_ts_anomalies['is_ts_anomaly']]
        ts_anomalies.to_csv(f"{self.output_dir}/time_series_anomalies.csv", index=False)
        
        print(f"Time series analysis complete. Found {len(ts_anomalies)} anomalies.")
        
        return data_with_ts_anomalies
    
    def compare_anomalies(self, stat_data, if_data, ts_data):
        """
        Compare anomalies detected by different methods and find consensus.
        
        Args:
            stat_data (DataFrame): Data with statistical outlier flags
            if_data (DataFrame): Data with Isolation Forest anomaly flags
            ts_data (DataFrame): Data with time series anomaly flags
            
        Returns:
            DataFrame: Combined data with consensus anomaly flags
        """
        # Create a combined dataset
        combined_data = self.data.copy()
        
        # Add flags from each method
        combined_data['is_stat_outlier'] = stat_data['is_outlier']
        combined_data['is_if_anomaly'] = if_data['is_anomaly_if']
        combined_data['is_ts_anomaly'] = ts_data['is_ts_anomaly']
        
        # Count how many methods flagged each point
        combined_data['anomaly_count'] = (
            combined_data['is_stat_outlier'].astype(int) +
            combined_data['is_if_anomaly'].astype(int) +
            combined_data['is_ts_anomaly'].astype(int)
        )
        
        # Flag as consensus anomaly if at least 2 methods agree
        combined_data['is_consensus_anomaly'] = combined_data['anomaly_count'] >= 2
        
        # Save consensus anomalies to CSV
        consensus_anomalies = combined_data[combined_data['is_consensus_anomaly']]
        consensus_anomalies.to_csv(f"{self.output_dir}/consensus_anomalies.csv", index=False)
        
        print(f"Consensus analysis complete. Found {len(consensus_anomalies)} consensus anomalies.")
        
        return combined_data
    
    def visualize_anomalies(self, combined_data):
        """
        Create visualizations of detected anomalies.
        
        Args:
            combined_data (DataFrame): Data with anomaly flags
        """
        # Plot anomalies by system-subsystem
        for (sys, subsys), group in combined_data.groupby(['system', 'subsystem']):
            if len(group) >= 5:  # Need enough data points
                plt.figure(figsize=(14, 6))
                
                # Sort by timestamp
                group = group.sort_values('timestamp')
                
                # Plot original data
                plt.plot(group['timestamp'], group['y_axis'], 'b-', label='Data')
                
                # Plot anomalies from different methods
                plt.scatter(
                    group[group['is_stat_outlier']]['timestamp'],
                    group[group['is_stat_outlier']]['y_axis'],
                    color='orange', marker='o', label='Statistical Outlier'
                )
                plt.scatter(
                    group[group['is_if_anomaly']]['timestamp'],
                    group[group['is_if_anomaly']]['y_axis'],
                    color='red', marker='^', label='Isolation Forest'
                )
                plt.scatter(
                    group[group['is_ts_anomaly']]['timestamp'],
                    group[group['is_ts_anomaly']]['y_axis'],
                    color='purple', marker='s', label='Time Series'
                )
                plt.scatter(
                    group[group['is_consensus_anomaly']]['timestamp'],
                    group[group['is_consensus_anomaly']]['y_axis'],
                    color='green', marker='*', s=200, label='Consensus'
                )
                
                plt.title(f'Anomaly Detection: {sys} - {subsys}')
                plt.xlabel('Timestamp')
                plt.ylabel('y_axis Value')
                plt.legend()
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(f"{self.output_dir}/anomalies_{sys}_{subsys}.png")
                plt.close()
        
        # Create summary plot of anomaly distribution across systems
        anomaly_summary = combined_data.groupby('system')['is_consensus_anomaly'].sum().reset_index()
        anomaly_summary.columns = ['system', 'anomaly_count']
        
        plt.figure(figsize=(12, 6))
        sns.barplot(x='system', y='anomaly_count', data=anomaly_summary)
        plt.title('Consensus Anomalies by System')
        plt.xlabel('System')
        plt.ylabel('Number of Anomalies')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/anomaly_summary_by_system.png")
        
    def run_full_analysis(self):
        """Run the complete anomaly detection workflow."""
        print("Starting aircraft systems anomaly detection analysis...")
        
        # Exploratory analysis
        print("\n1. Performing exploratory analysis...")
        system_stats = self.explore_data()
        print(system_stats)
        
        # Statistical outlier detection
        print("\n2. Detecting statistical outliers...")
        stat_data = self.detect_statistical_outliers(z_threshold=3.0)
        
        # Isolation Forest anomaly detection
        print("\n3. Detecting anomalies using Isolation Forest...")
        if_data = self.detect_anomalies_isolation_forest(contamination=0.05)
        
        # Time series anomaly detection
        print("\n4. Detecting time series anomalies...")
        ts_data = self.detect_time_series_anomalies(window_size=5, threshold=3.0)
        
        # Compare and combine results
        print("\n5. Finding consensus anomalies...")
        combined_data = self.compare_anomalies(stat_data, if_data, ts_data)
        
        # Visualize results
        print("\n6. Creating visualizations...")
        self.visualize_anomalies(combined_data)
        
        print(f"\nAnalysis complete! Results saved to {self.output_dir}/")
        
        return combined_data

# Example usage
if __name__ == "__main__":
    # File path to your data (update this!)
    data_path = "aircraft_data.csv"
    
    # Create detector
    detector = AircraftAnomalyDetector(data_path)
    
    # Run analysis
    results = detector.run_full_analysis()
    
    # Optional: analyze specific subsystem in detail
    # System3 had very high values in your sample
    # system3_data = results[results['system'] == 'System3']
    # if len(system3_data) > 0:
    #     print("\nSystem3 Anomalies Analysis:")
    #     print(system3_data[system3_data['is_consensus_anomaly']][['subsystem', 'timestamp', 'y_axis']])
