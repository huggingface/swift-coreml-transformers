//
//  CoreMLGPT2Tests.swift
//  CoreMLGPT2Tests
//
//  Created by Julien Chaumond on 18/07/2019.
//  Copyright © 2019 Hugging Face. All rights reserved.
//

import XCTest
@testable import CoreMLGPT2

struct EncodingSampleDataset: Decodable {
    let text: String
    let encoded_text: [String]
}

struct EncodingSample {
    static let dataset: EncodingSampleDataset = {
        let url = Bundle.main.url(forResource: "encoded_tokens", withExtension: "json")!
        let json = try! Data(contentsOf: url)
        let decoder = JSONDecoder()
        let dataset = try! decoder.decode(EncodingSampleDataset.self, from: json)
        return dataset
    }()
}



class CoreMLGPT2Tests: XCTestCase {

    override func setUp() {
        // Put setup code here. This method is called before the invocation of each test method in the class.
    }

    override func tearDown() {
        // Put teardown code here. This method is called after the invocation of each test method in the class.
    }

    func testByteEncode() {
        let dataset = EncodingSample.dataset
        
        let tokenizer = GPT2Tokenizer()
        XCTAssertEqual(
            tokenizer.byteEncode(text: dataset.text),
            dataset.encoded_text
        )
    }
}
