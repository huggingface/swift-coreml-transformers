# DistilBERT performance benchmarks

#### Full BERT-Squad with tokenization/featurization.

```
~/swift-coreml-transformers/CoreMLBertTests/BertForQATests.swift:75: 

Test Case '-[CoreMLBertTests.BertForQATests testPerformanceExample]' measured [Time, seconds] 
average: 1.583, relative standard deviation: 5.232%, values: [1.746976, 1.550390, 1.549479, 1.529654, 1.528065, 1.508825, 1.534357, 1.702786, 1.514816, 1.665434], performanceMetricID:com.apple.XCTPerformanceMetric_WallClockTime, baselineName: "", baselineAverage: , maxPercentRegression: 10.000%, maxPercentRelativeStandardDeviation: 10.000%, maxRegression: 0.100, maxStandardDeviation: 0.100

Test Case '-[CoreMLBertTests.BertForQATests testPerformanceExample]' passed (16.130 seconds).
```

---

#### Full BERT-Squad, only the inference.

```
~/swift-coreml-transformers/CoreMLBertTests/DistilBERTPerfTests.swift:23: 

Test Case '-[CoreMLBertTests.DistilBERTPerfTests testPerformanceNakedModel]' measured [Time, seconds] 
average: 1.118, relative standard deviation: 5.919%, values: [1.195310, 1.068182, 1.131890, 1.251984, 1.095551, 1.186633, 1.060465, 1.072363, 1.061609, 1.059508], performanceMetricID:com.apple.XCTPerformanceMetric_WallClockTime, baselineName: "", baselineAverage: , maxPercentRegression: 10.000%, maxPercentRelativeStandardDeviation: 10.000%, maxRegression: 0.100, maxStandardDeviation: 0.100

Test Case '-[CoreMLBertTests.DistilBERTPerfTests testPerformanceNakedModel]' passed (11.822 seconds).
```

---

#### DistilBERT, only the inference.

```
~/swift-coreml-transformers/CoreMLBertTests/DistilBERTPerfTests.swift:32: 

Test Case '-[CoreMLBertTests.DistilBERTPerfTests testPerformanceDistilBERTModel]' measured [Time, seconds] 
average: 0.319, relative standard deviation: 0.548%, values: [0.321627, 0.321694, 0.317964, 0.316413, 0.318463, 0.319897, 0.319386, 0.317997, 0.318780, 0.321835], performanceMetricID:com.apple.XCTPerformanceMetric_WallClockTime, baselineName: "", baselineAverage: , maxPercentRegression: 10.000%, maxPercentRelativeStandardDeviation: 10.000%, maxRegression: 0.100, maxStandardDeviation: 0.100

Test Case '-[CoreMLBertTests.DistilBERTPerfTests testPerformanceDistilBERTModel]' passed (3.466 seconds).
```

---
