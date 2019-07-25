//
//  MultiArrayUtilsTests.swift
//  CoreMLGPT2Tests
//
//  Created by Julien Chaumond on 25/07/2019.
//  Copyright © 2019 Hugging Face. All rights reserved.
//

import XCTest
import CoreML
@testable import CoreMLGPT2

class MultiArrayUtilsTests: XCTestCase {
    
    func testSlice() {
        let arr = MLMultiArray.testTensor(shape: [2, 3, 4])
        
        let o = MLMultiArray.slice(arr, sliceDim: 1, selectDims: [0: 0, 2: 0])
        print(o.debug)
        XCTAssertEqual(
            MLMultiArray.toDoubleArray(o),
            [2, 6, 10]
        )
        let oo = MLMultiArray.slice(arr, indexing: [.select(0), .slice, .select(0)])
        print(oo.debug)
        XCTAssertEqual(
            MLMultiArray.toDoubleArray(oo),
            [2, 6, 10]
        )
    }
    
    func testTopK() {
        let arr: [Double] = Array(0..<4).map { Double($0) } + Array(30..<36).map { Double($0) }
        XCTAssertEqual(
            arr,
            [0.0, 1.0, 2.0, 3.0, 30.0, 31.0, 32.0, 33.0, 34.0, 35.0]
        )
        let topk = Math.topK(arr: arr, k: 6)
        XCTAssertEqual(
            topk.indexes,
            [9, 8, 7, 6, 5, 4]
        )
        XCTAssertEqual(
            topk.probs,
            [0.6336913, 0.233122, 0.085760795, 0.031549633, 0.011606461, 0.0042697783]
        )
        
        let sampleIndex = Math.sample(indexes: topk.indexes, probs: topk.probs)
        print("sampleIndex", sampleIndex)
        
        XCTAssertTrue(
            sampleIndex >= 4 && sampleIndex <= 9
        )
    }
}
