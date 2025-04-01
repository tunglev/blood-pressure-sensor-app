// example.js
import React, { useState, useEffect } from 'react';
import { View, Text, Button, StyleSheet } from 'react-native';
import { SignalAnalyzer } from './backend/calculations';
import { SignalWrapper } from './wrappers';
import { LineChart } from 'react-native-chart-kit'; // You can use any chart library of your choice

// Generate a sample oscillometric signal
const generateSampleData = () => {
  // Create time array (4 seconds at 250Hz)
  const time = Array.from({ length: 1000 }, (_, i) => i / 250);
  
  // Create a sample oscillometric signal
  // Simulate a decreasing pressure with oscillations
  const baseSignal = time.map(t => 120 - 40 * (t / 4)); // Decreasing from 120 to 80
  const oscillations = time.map(t => {
    // Increase amplitude in the middle to simulate MAP area
    const amplitudeFactor = 1 + 6 * Math.exp(-Math.pow((t - 2) / 0.5, 2));
    return 5 * amplitudeFactor * Math.sin(2 * Math.PI * 1.2 * t);
  });
  
  // Combine the trend and oscillations
  const signal = baseSignal.map((val, idx) => val + oscillations[idx]);
  
  return { time, signal };
};

const BloodPressureAnalysis = () => {
  const [bpResults, setBpResults] = useState(null);
  const [data, setData] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);

  useEffect(() => {
    // Generate sample data on component mount
    setData(generateSampleData());
  }, []);

  const analyzeBP = () => {
    if (!data) return;
    
    setIsAnalyzing(true);
    
    // Simulate async processing (this would be important for real signals that might take time to process)
    setTimeout(() => {
      try {
        // Estimate blood pressure
        const { SBP, DBP } = SignalAnalyzer.estimatebp(data.time, data.signal);
        
        // Scale the normalized values to actual pressure range
        // In a real app, you would calibrate this based on your sensor
        const systolic = Math.round(SBP * 40 + 100); // Convert normalized to mmHg
        const diastolic = Math.round(DBP * 40 + 60); // Convert normalized to mmHg
        
        setBpResults({ systolic, diastolic });
      } catch (error) {
        console.error('Error analyzing blood pressure:', error);
        setBpResults({ error: 'Could not analyze signal' });
      } finally {
        setIsAnalyzing(false);
      }
    }, 500);
  };

  // Format data for the chart
  const getChartData = () => {
    if (!data) return null;
    
    // Downsample for display (showing every 8th point)
    const downsampledTime = data.time.filter((_, i) => i % 8 === 0);
    const downsampledSignal = data.signal.filter((_, i) => i % 8 === 0);
    
    return {
      labels: downsampledTime.map(t => t.toFixed(1)),
      datasets: [
        {
          data: downsampledSignal,
          color: () => 'rgba(75, 192, 192, 1)',
          strokeWidth: 2
        }
      ]
    };
  };

  return (
    <View style={styles.container}>
      <Text style={styles.title}>Blood Pressure Analyzer</Text>
      
      {data && (
        <View style={styles.chartContainer}>
          <Text style={styles.chartTitle}>Oscillometric Signal</Text>
          <LineChart
            data={getChartData()}
            width={350}
            height={220}
            chartConfig={{
              backgroundColor: '#ffffff',
              backgroundGradientFrom: '#ffffff',
              backgroundGradientTo: '#ffffff',
              decimalPlaces: 1,
              color: (opacity = 1) => `rgba(0, 0, 0, ${opacity})`,
              labelColor: (opacity = 1) => `rgba(0, 0, 0, ${opacity})`,
              style: {
                borderRadius: 16
              }
            }}
            bezier
            style={styles.chart}
          />
        </View>
      )}
      
      <Button
        title={isAnalyzing ? "Analyzing..." : "Analyze Blood Pressure"}
        onPress={analyzeBP}
        disabled={isAnalyzing || !data}
      />
      
      {bpResults && !bpResults.error && (
        <View style={styles.resultsContainer}>
          <Text style={styles.resultTitle}>Blood Pressure Results</Text>
          <Text style={styles.resultText}>
            {bpResults.systolic}/{bpResults.diastolic} mmHg
          </Text>
        </View>
      )}
      
      {bpResults && bpResults.error && (
        <Text style={styles.errorText}>{bpResults.error}</Text>
      )}
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    padding: 20,
    backgroundColor: '#f5f5f5'
  },
  title: {
    fontSize: 24,
    fontWeight: 'bold',
    marginBottom: 20
  },
  chartContainer: {
    marginBottom: 20,
    backgroundColor: 'white',
    borderRadius: 10,
    padding: 10,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 2,
    elevation: 2
  },
  chartTitle: {
    fontSize: 16,
    fontWeight: 'bold',
    textAlign: 'center',
    marginBottom: 10
  },
  chart: {
    borderRadius: 10
  },
  resultsContainer: {
    marginTop: 20,
    alignItems: 'center',
    backgroundColor: 'white',
    padding: 15,
    borderRadius: 10,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 2,
    elevation: 2
  },
  resultTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    marginBottom: 10
  },
  resultText: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#2c3e50'
  },
  errorText: {
    marginTop: 20,
    color: 'red',
    fontSize: 16
  }
});

export default BloodPressureAnalysis; 