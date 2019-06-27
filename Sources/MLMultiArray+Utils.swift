//
//  MLMultiArray+Utils.swift
//  CoreMLBert
//
//  Created by Julien Chaumond on 27/06/2019.
//  Copyright © 2019 Hugging Face. All rights reserved.
//

import Foundation
import CoreML

extension MLMultiArray {
    static func from(_ arr: [Int]) -> MLMultiArray {
        let o = try! MLMultiArray(shape: [1, arr.count] as [NSNumber], dataType: .int32)
        let ptr = UnsafeMutablePointer<Int32>(OpaquePointer(o.dataPointer))
        for (i, item) in arr.enumerated() {
            ptr[i] = Int32(item)
        }
        return o
    }
    
    /// This will concatenate all dimenions into one one-dim array.
    static func toIntArray(_ o: MLMultiArray) -> [Int] {
        var arr = Array(repeating: 0, count: o.count)
        let ptr = UnsafeMutablePointer<Int32>(OpaquePointer(o.dataPointer))
        for i in 0..<o.count {
            arr[i] = Int(ptr[i])
        }
        return arr
    }
}


extension MLMultiArray {
    var debug: String {
        return debug([])
    }
    
    /// From https://twitter.com/mhollemans
    ///
    /// Slightly tweaked
    ///
    func debug(_ indices: [Int]) -> String {
        func indent(_ x: Int) -> String {
            return String(repeating: " ", count: x)
        }
        
        // This function is called recursively for every dimension.
        // Add an entry for this dimension to the end of the array.
        var indices = indices + [0]
        
        let d = indices.count - 1          // the current dimension
        let N = shape[d].intValue          // how many elements in this dimension
        var s = "["
        if indices.count < shape.count {   // not last dimension yet?
            for i in 0..<N {
                indices[d] = i
                s += debug(indices)        // then call recursively again
                if i != N - 1 {
                    s += ",\n" + indent(d + 1)
                }
            }
        } else {                           // the last dimension has actual data
            s += " "
            for i in 0..<N {
                indices[d] = i
                s += "\(self[indices as [NSNumber]])"
                if i != N - 1 {                // not last element?
                    s += ", "
                    if i % 11 == 10 {            // wrap long lines
                        s += "\n " + indent(d + 1)
                    }
                }
            }
            s += " "
        }
        return s + "]"
    }
}
