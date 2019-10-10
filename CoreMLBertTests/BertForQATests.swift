//
//  BertForQATests.swift
//  CoreMLBertTests
//
//  Created by Julien Chaumond on 27/06/2019.
//  Copyright © 2019 Hugging Face. All rights reserved.
//

import XCTest
import CoreML
@testable import CoreMLBert


class BertForQATests: XCTestCase {
    
    let context = "Super Bowl 50 was an American football game to determine the champion of the National Football League (NFL) for the 2015 season. The American Football Conference (AFC) champion Denver Broncos defeated the National Football Conference (NFC) champion Carolina Panthers 24–10 to earn their third Super Bowl title. The game was played on February 7, 2016, at Levi's Stadium in the San Francisco Bay Area at Santa Clara, California. As this was the 50th Super Bowl, the league emphasized the \"golden anniversary\" with various gold-themed initiatives, as well as temporarily suspending the tradition of naming each Super Bowl game with Roman numerals (under which the game would have been known as \"Super Bowl L\"), so that the logo could prominently feature the Arabic numerals 50."
    
    let question = "Which NFL team represented the AFC at Super Bowl 50?"
    let m = BertForQuestionAnswering(.full)
    
    func testFeaturizationLength() {
        let tokenizer = BertTokenizer()
        
        var nTooLong = 0
        for example in Squad.examples {
            let tokensQuestion = tokenizer.tokenize(text: example.question)
            let tokensContext = tokenizer.tokenize(text: example.context)
            if tokensQuestion.count + tokensContext.count + 3 > m.seqLen {
                nTooLong += 1
            }
        }
        print(
            "🍌 Too long: \(nTooLong)/\(Squad.examples.count)"
        )
    }
    
    func testFeaturizationFull() {
        
        let input = m.featurizeTokens(question: question, context: context)
        let word_id = [101, 2029, 5088, 2136, 3421, 1996, 10511, 2012, 3565, 4605, 2753, 1029, 102, 3565, 4605, 2753, 2001, 2019, 2137, 2374, 2208, 2000, 5646, 1996, 3410, 1997, 1996, 2120, 2374, 2223, 1006, 5088, 1007, 2005, 1996, 2325, 2161, 1012, 1996, 2137, 2374, 3034, 1006, 10511, 1007, 3410, 7573, 14169, 3249, 1996, 2120, 2374, 3034, 1006, 22309, 1007, 3410, 3792, 12915, 2484, 1516, 2184, 2000, 7796, 2037, 2353, 3565, 4605, 2516, 1012, 1996, 2208, 2001, 2209, 2006, 2337, 1021, 1010, 2355, 1010, 2012, 11902, 1005, 1055, 3346, 1999, 1996, 2624, 3799, 3016, 2181, 2012, 4203, 10254, 1010, 2662, 1012, 2004, 2023, 2001, 1996, 12951, 3565, 4605, 1010, 1996, 2223, 13155, 1996, 1000, 3585, 5315, 1000, 2007, 2536, 2751, 1011, 11773, 11107, 1010, 2004, 2092, 2004, 8184, 28324, 2075, 1996, 4535, 1997, 10324, 2169, 3565, 4605, 2208, 2007, 3142, 16371, 28990, 2015, 1006, 2104, 2029, 1996, 2208, 2052, 2031, 2042, 2124, 2004, 1000, 3565, 4605, 1048, 1000, 1007, 1010, 2061, 2008, 1996, 8154, 2071, 14500, 3444, 1996, 5640, 16371, 28990, 2015, 2753, 1012, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        let word_type = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        
        XCTAssertEqual(
            MLMultiArray.toIntArray(input.word_id), word_id
        )
        XCTAssertEqual(
            MLMultiArray.toIntArray(input.word_type), word_type
        )
    }
    
    func testPredict() {
        let prediction = m.predict(question: question, context: context)
        XCTAssertEqual(
            prediction.start, 46
        )
        XCTAssertEqual(
            prediction.end, 47
        )
        XCTAssertEqual(
            prediction.tokens, ["denver", "broncos"]
        )
        XCTAssertEqual(
            prediction.answer, "denver broncos"
        )
    }
    
    
    func testPerformanceFull() {
        /// The whole model + tokenizer + argmax
        // 1.646 seconds average for inference in Simulator.
        self.measure {
            _ = m.predict(question: question, context: context)
        }
    }
    
    func testPerformanceNaked() {
        /// Only the model's forward pass
        // average: 1.104 s in Simulator
        // average: 0.472 s on device
        let input = m.featurizeTokens(question: question, context: context)
        
        guard case .full(let model) = m.model else {
            return
        }
        
        self.measure {
            _ = try! model.prediction(input: input)
        }
    }
}
