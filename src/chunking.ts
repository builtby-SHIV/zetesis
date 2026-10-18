import { GoogleGenAI } from "@google/genai";
import { randomUUID } from "node:crypto";

export interface Chunk {
  id: string;
  text: string;
  type: "parent" | "child";
  parentId?: string;
  metaData?: Record<string, any>;
  tokens: number;
}

export async function getTokenCount(text: string, ai: GoogleGenAI) {
    try {
        const res = await ai.models.countTokens({
        model: "gemini-flash-latest",
        contents: text,
        });

        return res;
    } catch (e) {
        console.error(e);
    }
}

export async function chunkDoc(rawText: string, metaData: Record<string, any> = {}) {
    const allChunks: Chunk[] = [];
    const parentTexts = rawText.split(/\n\n+/).filter((p) => p.trim().length > 0);

    for (const text of parentTexts) {
        const parentChunk: Chunk = {
            id: randomUUID(),
            text: text.trim(),
            type: 'parent',
            metaData,
            tokens: 0,
        };
        allChunks.push(parentChunk);
    }

    return allChunks;
}
