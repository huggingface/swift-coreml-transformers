//
//  DistilbertForQATests.swift
//  CoreMLBertTests
//
//  Created by Julien Chaumond on 09/10/2019.
//  Copyright © 2019 Hugging Face. All rights reserved.
//

import XCTest

class DistilbertForQATests: XCTestCase {

    let context = "Super Bowl 50 was an American football game to determine the champion of the National Football League (NFL) for the 2015 season. The American Football Conference (AFC) champion Denver Broncos defeated the National Football Conference (NFC) champion Carolina Panthers 24–10 to earn their third Super Bowl title. The game was played on February 7, 2016, at Levi's Stadium in the San Francisco Bay Area at Santa Clara, California. As this was the 50th Super Bowl, the league emphasized the \"golden anniversary\" with various gold-themed initiatives, as well as temporarily suspending the tradition of naming each Super Bowl game with Roman numerals (under which the game would have been known as \"Super Bowl L\"), so that the logo could prominently feature the Arabic numerals 50."
    
    let question = "Which NFL team represented the AFC at Super Bowl 50?"
    let m = BertForQuestionAnswering(.distilled)

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
        // 0.942 seconds average in Simulator
        self.measure {
            _ = m.predict(question: question, context: context)
        }
    }
    
    func testPerformanceNaked() {
        /// Only the model's forward pass
        // average: 0.878 s in Simulator
        // average: 0.311 s on device
        let input = m.featurizeTokensDistilled(question: question, context: context)
        
        guard case .distilled(let model) = m.model else {
            return
        }
        
        self.measure {
            _ = try! model.prediction(input: input)
        }
    }
}
