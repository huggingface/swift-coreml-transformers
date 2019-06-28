//
//  BertTokenizerTests.swift
//  CoreMLBertTests
//
//  Created by Julien Chaumond on 27/06/2019.
//  Copyright © 2019 Hugging Face. All rights reserved.
//

import XCTest
@testable import CoreMLBert



class BertTokenizerTests: XCTestCase {

    override func setUp() {
        // Put setup code here. This method is called before the invocation of each test method in the class.
    }

    override func tearDown() {
        // Put teardown code here. This method is called after the invocation of each test method in the class.
    }

    func testBasicTokenizer() {
        let basicTokenizer = BasicTokenizer()
        
        let text = "Brave gaillard, d'où [UNK] êtes vous?"
        let tokens = ["brave", "gaillard", ",", "d", "\'", "ou", "[UNK]", "etes", "vous", "?"]
        
        XCTAssertEqual(
            basicTokenizer.tokenize(text: text), tokens
        )
        /// Verify that `XCTAssertEqual` does what deep equality checks on arrays of strings.
        XCTAssertEqual(["foo", "bar"], ["foo", "bar"])
    }
    
    /// For each Squad question tokenized by python, check that we get the same output through the `BasicTokenizer`
    func testFullBasicTokenizer() {
        let url = Bundle.main.url(forResource: "basic_tokenized_questions", withExtension: "json")!
        let json = try! Data(contentsOf: url)
        let decoder = JSONDecoder()
        let sampleTokens = try! decoder.decode([[String]].self, from: json)
        
        let basicTokenizer = BasicTokenizer()
        
        XCTAssertEqual(sampleTokens.count, Squad.examples.count)
        
        for (i, example) in Squad.examples.enumerated() {
            let output = basicTokenizer.tokenize(text: example.question)
            XCTAssertEqual(output, sampleTokens[i])
        }
    }
    
    /// For each Squad question tokenized by python, check that we get the same output through the whole `BertTokenizer`
    func testFullBertTokenizer() {
        let url = Bundle.main.url(forResource: "tokenized_questions", withExtension: "json")!
        let json = try! Data(contentsOf: url)
        let decoder = JSONDecoder()
        let sampleTokens = try! decoder.decode([[Int]].self, from: json)
        
        let tokenizer = BertTokenizer()
        
        XCTAssertEqual(sampleTokens.count, Squad.examples.count)
        
        for (i, example) in Squad.examples.enumerated() {
            let output = tokenizer.tokenizeToIds(text: example.question)
            XCTAssertEqual(output, sampleTokens[i])
        }
    }
    
    func testPerformanceExample() {
        let tokenizer = BertTokenizer()
        
        // This is an example of a performance test case.
        self.measure {
            // Put the code you want to measure the time of here.
            _ = tokenizer.tokenizeToIds(text: "Brave gaillard, d'où [UNK] êtes vous?")
        }
    }
    
    func testWordpieceDetokenizer() {
        struct QuestionTokens: Codable {
            let original: String
            let basic: [String]
            let wordpiece: [String]
        }
        
        let url = Bundle.main.url(forResource: "question_tokens", withExtension: "json")!
        let json = try! Data(contentsOf: url)
        let decoder = JSONDecoder()
        let questionTokens = try! decoder.decode([QuestionTokens].self, from: json)
        
        let tokenizer = BertTokenizer()
        
        for question in questionTokens {
            XCTAssertEqual(question.basic.joined(separator: " "), tokenizer.convertWordpieceToBasicTokenList(question.wordpiece))
        }
    }
}
