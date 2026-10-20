import 'dotenv/config';
import { parseDocument } from 'docmarrow';
import { readFile } from 'node:fs/promises';
import { GoogleGenAI } from '@google/genai';
import {
    BGE_MODEL_LIMIT,
    DEFAULT_MAX_CHILD_TOKENS,
    DEFAULT_MAX_PARENT_TOKENS,
    chunkDoc,
    getHierarchicalView,
    getTokenCount,
    type Chunk,
    type ChunkingOptions,
} from './chunking.ts';

export interface ProcessPdfResult {
    markdown: string;
    chunks: Chunk[];
    parents: Chunk[];
    children: Chunk[];
    hierarchicalView: Array<{ parent: Chunk; children: Chunk[] }>;
    stats: {
        pageCount: number;
        parentCount: number;
        childCount: number;
        avgParentTokens: number;
        avgChildTokens: number;
    };
}

export async function processPdfDocument(filePath: string): Promise<ProcessPdfResult> {
    const bytes = new Uint8Array(await readFile(filePath));

    const doc = await parseDocument(bytes, {
        format: 'pdf',
        tables: false,            
        readingOrder: true,       
        dropHeadersFooters: true, 
        speakerNotes: false,
    });

    const rawMarkdown = doc.markdown || '';

    const chunks = await chunkDoc(rawMarkdown, { source: filePath, pageCount: doc.meta?.pageCount });

    const parents = chunks.filter((c) => c.type === 'parent');
    const children = chunks.filter((c) => c.type === 'child');
    const hierarchicalView = getHierarchicalView(chunks);

    const totalParentTokens = parents.reduce((sum, p) => sum + p.tokens, 0);
    const totalChildTokens = children.reduce((sum, c) => sum + c.tokens, 0);

    const stats = {
        pageCount: doc.meta?.pageCount ?? 0,
        parentCount: parents.length,
        childCount: children.length,
        avgParentTokens: parents.length > 0 ? Math.round(totalParentTokens / parents.length) : 0,
        avgChildTokens: children.length > 0 ? Math.round(totalChildTokens / children.length) : 0,
    };

    return {
        markdown: rawMarkdown,
        chunks,
        parents,
        children,
        hierarchicalView,
        stats,
    };
}

async function main() {
    const filePath = process.argv[2] || 'sample3.pdf';
    console.log(`\n======================================================`);
    console.log(`Processing PDF: ${filePath}`);
    console.log(`Tokenizer: BAAI/bge-small-en-v1.5 (Context Limit: ${BGE_MODEL_LIMIT})`);
    console.log(`Targets:   Parent = ${DEFAULT_MAX_PARENT_TOKENS} tokens, Child = ${DEFAULT_MAX_CHILD_TOKENS} tokens`);
    console.log(`======================================================`);

    try {
        const result = await processPdfDocument(filePath);

        // console.log(`\nParsing & Hierarchical Chunking Summary:`);
        // console.log(`- Page Count:       ${result.stats.pageCount}`);
        // console.log(`- Parent Chunks:    ${result.stats.parentCount} (Avg Tokens: ${result.stats.avgParentTokens}, Target: ${DEFAULT_MAX_PARENT_TOKENS})`);
        // console.log(`- Child Chunks:     ${result.stats.childCount} (Avg Tokens: ${result.stats.avgChildTokens}, Max Target: ${DEFAULT_MAX_CHILD_TOKENS} <= ${BGE_MODEL_LIMIT})`);
        // console.log(`- Total Chunks:     ${result.chunks.length}`);

        // const sampleParent = result.parents[0];
        // if (sampleParent) {
        //     const sampleChildren = result.children.filter((c) => c.parentId === sampleParent.id);

        //     console.log(`\n------------------------------------------------------`);
        //     console.log(`[Sample Parent Chunk 0]`);
        //     console.log(`ID:      ${sampleParent.id}`);
        //     console.log(`Tokens:  ${sampleParent.tokens} (bge-small tokens)`);
        //     console.log(`Children:${sampleChildren.length}`);
        //     console.log(`Preview:\n${sampleParent.text.slice(0, 300)}...`);

        //     console.log(`\n[Child Chunks for Parent 0]`);
        //     sampleChildren.slice(0, 3).forEach((child, idx) => {
        //         console.log(`  Child ${idx + 1}/${sampleChildren.length} [Tokens: ${child.tokens} / max ${DEFAULT_MAX_CHILD_TOKENS}, ID: ${child.id.slice(0, 8)}...]`);
        //         console.log(`  Text: "${child.text.replace(/\n/g, ' ')}"`);
        //     });
        //     if (sampleChildren.length > 3) {
        //         console.log(`  ... and ${sampleChildren.length - 3} more child chunks`);
        //     }
        //     console.log(`------------------------------------------------------\n`);
        //     }

            const firstChunk = result.chunks[0];
            if (process.env.GEMINI_KEY && firstChunk) {
            try {
                const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_KEY });
                const sampleText = firstChunk.text.slice(0, 200);
                const geminiTokens = await getTokenCount(sampleText, ai);
                if (geminiTokens !== undefined) {
                    console.log(`Gemini API Token Check: Verified successfully (${geminiTokens} tokens for sample preview).\n`);
                }
            } catch {
                // Soft fail if API key is invalid or network issues occur
            }
        }
    } catch (error) {
        console.error(`Failed to process ${filePath}:`, error);
    }
}

main();