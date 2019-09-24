//
//  DistilBERTPerfTests.swift
//  CoreMLBertTests
//
//  Created by Julien Chaumond on 16/09/2019.
//  Copyright © 2019 Hugging Face. All rights reserved.
//

import XCTest
import CoreML
@testable import CoreMLBert

class DistilBERTPerfTests: XCTestCase {
    let context = "Super Bowl 50 was an American football game to determine the champion of the National Football League (NFL) for the 2015 season. The American Football Conference (AFC) champion Denver Broncos defeated the National Football Conference (NFC) champion Carolina Panthers 24–10 to earn their third Super Bowl title. The game was played on February 7, 2016, at Levi's Stadium in the San Francisco Bay Area at Santa Clara, California. As this was the 50th Super Bowl, the league emphasized the \"golden anniversary\" with various gold-themed initiatives, as well as temporarily suspending the tradition of naming each Super Bowl game with Roman numerals (under which the game would have been known as \"Super Bowl L\"), so that the logo could prominently feature the Arabic numerals 50."
    
    let question = "Which NFL team represented the AFC at Super Bowl 50?"
    let m = BertForQuestionAnswering()
    let mDistilbert = distilbert_64_12()
    let mDistilbert128 = distilbert_squad_128()
    
    func testPerformanceNakedBERTModel() {
        let input = m.featurizeTokens(question: question, context: context)
        
        self.measure {
            _ = try! m.model.prediction(input: input)
        }
    }
    
    func testPerformanceDistilBERTModel() {
        let input_ids = MLMultiArray.from(Array(repeating: 0, count: 64))
        
        self.measure {
            _ = try! mDistilbert.prediction(input_ids: input_ids)
            /// print(output.output_logits)
        }
    }
    
    func testPerformanceDistilbert128() {
        let input_ids = MLMultiArray.from(Array(repeating: 0, count: 128), dims: 2)
        
        self.measure {
            _ = try! mDistilbert128.prediction(input_ids: input_ids)
            /// print(output.output_logits)
        }
    }
}

