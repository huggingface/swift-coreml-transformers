//
//  CLIPTokenizer.swift
//  CoreMLBert
//
//  Created by Matthew Waller on 1/31/23.
//  Copyright © 2023 Hugging Face. All rights reserved.
//

import Foundation

class CLIPTokenizer {
    let bpeRanks: Dictionary<BytePair, Int>
    private let encoder: [String: Int]
    private let decoder: [Int: String]
    
    init() {
        let url = Bundle.main.url(forResource: "merges", withExtension: "txt")!
        let bpeMergesTxt = try! String(contentsOf: url)
        let arr = bpeMergesTxt.split(separator: "\n").map { String($0) }
        var bpeRanks: Dictionary<BytePair, Int> = [:]
        for i in 1..<arr.count {
            let tuple = arr[i].split(separator: " ").map { String($0) }
            let bp = BytePair(tuple: tuple)
            bpeRanks[bp] = i - 1
        }
        self.bpeRanks = bpeRanks
        
        self.encoder = {
            let url = Bundle.main.url(forResource: "vocab", withExtension: "json")!
            let json = try! Data(contentsOf: url)
            let decoder = JSONDecoder()
            let vocab = try! decoder.decode([String: Int].self, from: json)
            return vocab
        }()
        self.decoder = Utils.invert(self.encoder)
    }
    
    func byteEncode(text: String) -> [String] {
        let RE = "<\\|startoftext\\|>|<\\|endoftext\\|>|'s|'t|'re|'ve|'m|'ll|'d|[\\p{L}]+|[\\p{N}]|[^\\s\\p{L}\\p{N}]+"
        let tokens = text.ranges(of: RE).map { String(text[$0]) }
        return tokens.map { (token) -> String in
            return Array(token.utf8).map { byteEncoder[$0]! }.joined()
        }
    }
    
    private func getPairs(word: [String]) -> Set<BytePair> {
        var s = Set<BytePair>()
        for i in 0..<word.count-1 {
            let bp = BytePair(
                word[i],
                word[i+1]
            )
            s.insert(bp)
        }
        return s
    }
    
    func bpe(token: String) -> String {
        if token.count <= 1 {
            return token + "</w>"
        }
        
        var word = Array(token).map { String($0)}
        let last = (word.last ?? "") + "</w>"
        word.removeLast()
        word.append(last)
        var pairs = Array(getPairs(word: word))
        if pairs.isEmpty {
            return token + "</w>"
        }
        
        while true {
            let bigrams = pairs.filter { (bp) -> Bool in bpeRanks[bp] != nil }
            if bigrams.count == 0 {
                break
            }
            let bigram = bigrams.min { (bp1, bp2) -> Bool in
                return bpeRanks[bp1]! < bpeRanks[bp2]!
            }!
            let first = bigram.a
            let second = bigram.b
            var newWord: [String] = []
            var i = 0
            while i < word.count {
                if let j = word[i..<word.count].firstIndex(of: first) {
                    newWord.append(contentsOf: word[i..<j])
                    i = j
                } else {
                    newWord.append(contentsOf: word[i..<word.count])
                    break
                }
                
                if word[i] == first && i < word.count - 1 && word[i+1] == second {
                    newWord.append(first+second)
                    i += 2
                } else {
                    newWord.append(word[i])
                    i += 1
                }
            }
            word = newWord
            if word.count == 1 {
                break
            } else {
                pairs = Array(getPairs(word: word))
            }
        }
        return word.joined(separator: " ")
    }
    
    func tokenize(text: String) -> [String] {
        var tokens: [String] = []
        let lowercased = text.lowercased()
        for token in self.byteEncode(text: lowercased) {
            let xx = self.bpe(token: token).split(separator: " ").map { String($0) }
            tokens.append(contentsOf: xx)
        }
        return tokens
    }
    
    /// Main entry point
    func encode(text: String) -> [Int] {
        return tokenize(text: text).map { encoder[$0]! }
    }
    
    /// Decode
    func decode(tokens: [Int]) -> String {
        let text = tokens.map { decoder[$0]! }.joined(separator: "")
        let utfCodepoints = text.map { byteDecoder[String($0)]! }
        return String(decoding: utfCodepoints, as: UTF8.self)
    }
}
