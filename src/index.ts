import { GoogleGenAI } from "@google/genai";
import { parseDocument } from "docmarrow";
import { readFile } from "node:fs/promises";
import { getTokenCount } from "./chunking.ts";

async function parseDoc() {
    const bytes = new Uint8Array(await readFile('sample1.pdf'));
    const doc = await parseDocument(bytes, {
        format: "pdf",            
        tables: true,             
        readingOrder: true,       
        dropHeadersFooters: true, 
        password: "…",            
        speakerNotes: false,      
    });

    console.log(doc.markdown[9])

    if (process.env.GEMINI_KEY) {
        const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_KEY });
        getTokenCount(text, ai)
    }
}

parseDoc()