//
//  BertForQuestionAnswering.swift
//  CoreMLBert
//
//  Created by Julien Chaumond on 27/06/2019.
//  Copyright © 2019 Hugging Face. All rights reserved.
//

import Foundation
import CoreML

class BertForQuestionAnswering {
    private let model = BERTSQUADFP16()
    private let tokenizer = BertTokenizer()
    public let seqLen = 384
    
    
    /// Main prediction loop:
    /// - featurization
    /// - model inference
    /// - argmax and re-tokenization
    func predict(question: String, context: String) -> (start: Int, end: Int, tokens: [String], answer: String) {
        let input = featurizeTokens(question: question, context: context)
        
        let output = try! model.prediction(input: input)
        let start = Math.argmax(output.start_logits).0
        let end = Math.argmax(output.end_logits).0
        
        let tokenIds = Array(
            MLMultiArray.toIntArray(input.word_id)[start...end]
        )
        let tokens = tokenizer.unTokenize(tokens: tokenIds)
        let answer = tokenizer.convertWordpieceToBasicTokenList(tokens)
        return (start: start, end: end, tokens: tokens, answer: answer)
    }
    
    
    func featurizeTokens(question: String, context: String) -> BERTSQUADFP16Input {
        let tokensQuestion = tokenizer.tokenizeToIds(text: question)
        var tokensContext = tokenizer.tokenizeToIds(text: context)
        if tokensQuestion.count + tokensContext.count + 3 > seqLen {
            /// This case is fairly rare in the dev set (183/10570),
            /// so we just keep only the start of the context in that case.
            /// In Python, we use a more complex sliding window approach.
            /// see `pytorch-transformers`.
            let toRemove = tokensQuestion.count + tokensContext.count + 3 - seqLen
            tokensContext.removeLast(toRemove)
        }
        
        let nPadding = seqLen - tokensQuestion.count - tokensContext.count - 3
        /// Sequence of input symbols. The sequence starts with a start token (101) followed by question tokens that are followed be a separator token (102) and the document tokens.The document tokens end with a separator token (102) and the sequenceis padded with 0 values to length 384.
        var allTokens: [Int] = []
        /// Would love to create it in a single Swift line but Xcode compiler fails...
        allTokens.append(
            tokenizer.tokenToId(token: "[CLS]")
        )
        allTokens.append(contentsOf: tokensQuestion)
        allTokens.append(
            tokenizer.tokenToId(token: "[SEP]")
        )
        allTokens.append(contentsOf: tokensContext)
        allTokens.append(
            tokenizer.tokenToId(token: "[SEP]")
        )
        allTokens.append(contentsOf: Array(repeating: 0, count: nPadding))
        let word_id = MLMultiArray.from(allTokens, dims: 2)
        
        /// Sequence of token-types. Values of 0 for the start token, question tokens and the question separator. Value 1 for the document tokens and the end separator. The sequence is padded with 0 values to length 384.
        var tokenTypes = Array(repeating: 0, count: seqLen)
        let startPos = 2 + tokensQuestion.count
        for i in startPos...startPos+tokensContext.count {
            tokenTypes[i] = 1
        }
        let word_type = MLMultiArray.from(tokenTypes, dims: 2)
        
        /// Fixed sequence of values from 0 to 383.
        let position = MLMultiArray.from(
            Array(0..<seqLen),
            dims: 2
        )
        
        /// A masking matrix (logits). It has zero values in the first X number of columns, where X = number of input tokens without the padding,and value -1e+4 in the remaining 384-X (padding) columns.
        let attention_mask = try! MLMultiArray(shape: [1, 1, 384, 384], dataType: .double)
        for i in 0..<384 {
            for j in 0..<384 {
                attention_mask[[0, 0, i, j] as [NSNumber]] = (j < tokensQuestion.count + tokensContext.count + 3) ? 0 : -1_000
            }
        }
        
        return BERTSQUADFP16Input(word_id: word_id, word_type: word_type, position: position, attention_mask: attention_mask)
    }
}
