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
        
        let oo = MLMultiArray.slice(arr, indexing: [.select(0), .slice, .select(0)])
        print(oo.debug)
    }
}
