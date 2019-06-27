//
//  Math.swift
//  CoreMLBert
//
//  Created by Julien Chaumond on 27/06/2019.
//  Copyright © 2019 Hugging Face. All rights reserved.
//

import Foundation
import Accelerate

///
/// From M.I. Hollemans
///
/// https://github.com/hollance/CoreMLHelpers
///
struct Math {
    
    /**
     Returns the index and value of the largest element in the array.
     
     - Parameters:
     - ptr: Pointer to the first element in memory.
     - count: How many elements to look at.
     - stride: The distance between two elements in memory.
     */
    static func argmax(_ ptr: UnsafePointer<Float>, count: Int, stride: Int = 1) -> (Int, Float) {
        var maxValue: Float = 0
        var maxIndex: vDSP_Length = 0
        vDSP_maxvi(ptr, vDSP_Stride(stride), &maxValue, &maxIndex, vDSP_Length(count))
        return (Int(maxIndex), maxValue)
    }
}
