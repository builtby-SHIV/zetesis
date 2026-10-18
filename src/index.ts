import { parseDocument } from "docmarrow";
import { readFile } from "node:fs/promises";

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
}

parseDoc()