import { AutoTokenizer, PreTrainedTokenizer } from '@xenova/transformers';
import { GoogleGenAI } from '@google/genai';
import { randomUUID } from 'node:crypto';

export const BGE_MODEL_LIMIT = 512;
export const DEFAULT_MAX_PARENT_TOKENS = 1000;
export const DEFAULT_MAX_CHILD_TOKENS = 400;

export interface Chunk {
    id: string;
    text: string;
    type: 'parent' | 'child';
    parentId?: string;
    metaData: Record<string, any>;
    tokens: number;
}

export interface ChunkingOptions {
    maxParentTokens?: number;
    maxChildTokens?: number;
    tokenizer?: PreTrainedTokenizer;
}

let cachedTokenizer: PreTrainedTokenizer | null = null;
let tokenizerPromise: Promise<PreTrainedTokenizer> | null = null;

export async function getBgeTokenizer(): Promise<PreTrainedTokenizer> {
    if (cachedTokenizer) 
        return cachedTokenizer;
    if (!tokenizerPromise) {
        tokenizerPromise = AutoTokenizer.from_pretrained('Xenova/bge-small-en-v1.5').then(
            (tokenizer: any) => {
                cachedTokenizer = tokenizer as PreTrainedTokenizer;
                return cachedTokenizer;
            }
        );
    }
    return tokenizerPromise;
}

export function countTokens(text: string, tokenizer?: PreTrainedTokenizer | null): number {
    if (!text || !text.trim()) 
        return 0;
    const t = tokenizer ?? cachedTokenizer;
    if (t)
        return t.encode(text).length;
    return Math.ceil(text.length / 4);
}

export async function countTokensWithBge(text: string): Promise<number> {
    const tokenizer = await getBgeTokenizer();
    return countTokens(text, tokenizer);
}

export async function getTokenCount(text: string, ai: GoogleGenAI): Promise<number | undefined> {
    try {
        const res = await ai.models.countTokens({
            model: 'gemini-flash-latest',
            contents: text,
        });
        return res.totalTokens;
    } catch (e) {
        console.error('Error counting tokens with Gemini API:', e);
        return undefined;
    }
}

const segmenter = new Intl.Segmenter('en', { granularity: 'sentence' });

export function splitIntoSentences(text: string): string[] {
    if (!text)
        return [];
    return Array.from(segmenter.segment(text), (s) => s.segment.trim()).filter(Boolean);
}

export function isListItem(line: string): boolean {
    const trimmed = line.trim();
    return /^(?:[•◦▪▫‣⁃∙*+-]|\d+[\.\)]|\([0-9a-zA-Z]{1,3}\)|[a-zA-Z][\.\)])\s+/.test(trimmed);
}

export function isHeading(line: string): boolean {
    const trimmed = line.trim();
    return /^#{1,6}\s+/.test(trimmed);
}

export function cleanPdfText(rawText: string): string {
    if (!rawText) 
        return '';

    const stripped = rawText
        .replace(/!\[.*?\](?:\(.*?\)|\[.*?\])?/g, '')
        .replace(/^\s*\|.*\|\s*$/gm, '')
        .replace(/^(?:[-*_]\s*){3,}$/gm, '');

    const paragraphs = stripped.split(/\r?\n\s*\r?\n+/);

    const cleaned = paragraphs.map((para) => {
        const lines = para.split(/\r?\n/).map((l) => l.trim()).filter(Boolean);
        if (!lines.length) 
            return '';

        let result = '';
        let lastWasHeading = false;

        for (const line of lines) {
            const isBullet = isListItem(line);
            const isHead = isHeading(line);

            if (!result)
                result = line;
            else if (isBullet || isHead || lastWasHeading)
                result += '\n' + line;
            else if (result.endsWith('-') && /^[a-z]/.test(line))
                result = result.slice(0, -1) + line;
            else 
                result += ' ' + line;

            lastWasHeading = isHead;
        }
        return result;
    });

    return cleaned.filter(Boolean).join('\n\n');
}

function extractChildUnits(text: string) {
    const blocks = text.split(/\n\n+/).filter(Boolean);
    
    const units: Array<{ text: string; isList: boolean }> = [];

    for (const block of blocks) {
        const lines = block.split(/\n/).map((l) => l.trim()).filter(Boolean);
        let currentTextLines: string[] = [];
        let currentListLines: string[] = [];

        function flushText() {
            if (currentTextLines.length === 0) 
                return;
            const joined = currentTextLines.join(' ');
            // console.log(joined)
            const sents = splitIntoSentences(joined);
            for (const s of sents) {
                units.push({ text: s, isList: false });
            }
            currentTextLines = [];
        }

        function flushList() {
            if (currentListLines.length === 0) 
                return;
            units.push({ text: currentListLines.join('\n'), isList: true });
            currentListLines = [];
        }

        for (const line of lines) {
            if (isListItem(line)) {
                flushText();
                currentListLines.push(line);
            } else {
                flushList();
                currentTextLines.push(line);
            }
        }

        flushText();
        flushList();
    }

    return units;
}

export async function chunkDoc(rawText: string, metaData: Record<string, any> = {}) {
    const tokenizer = await getBgeTokenizer();
    const maxParentTokens = DEFAULT_MAX_PARENT_TOKENS;
    const maxChildTokens = Math.min(DEFAULT_MAX_CHILD_TOKENS, BGE_MODEL_LIMIT);

    if (DEFAULT_MAX_CHILD_TOKENS > BGE_MODEL_LIMIT) {
        console.warn(
        `[Warning] requested maxChildTokens (${DEFAULT_MAX_CHILD_TOKENS}) exceeds bge-small 512 token limit. Clamping to ${BGE_MODEL_LIMIT}.`
        );
    }

    const cleanedText = cleanPdfText(rawText);
    if (!cleanedText) return [];

    const blocks = cleanedText.split(/\n\n+/).filter(Boolean);
    if (blocks.length === 0) return [];

    const parentGroups: string[] = [];
    let currentParent = '';

    function flushParent() {
        if (currentParent.trim()) {
            parentGroups.push(currentParent.trim());
            currentParent = '';
        }
    }

    for (const block of blocks) {
        const blockTokens = countTokens(block, tokenizer);
        console.log(currentParent)
        if (blockTokens > maxParentTokens) {
            flushParent();
            const subUnits = block.split(/\n/).filter(Boolean);
            for (const sub of subUnits) {
                const cand = currentParent ? `${currentParent}\n\n${sub}` : sub;
                if (countTokens(cand, tokenizer) > maxParentTokens && currentParent) {
                    flushParent();
                    currentParent = sub;
                } else {
                    currentParent = cand;
                }
            }
        } else {
            const cand = currentParent ? `${currentParent}\n\n${block}` : block;
            if (countTokens(cand, tokenizer) > maxParentTokens && currentParent) {
                flushParent();
                currentParent = block;
            } else {
                currentParent = cand;
            }
        }
    }
    flushParent();

    const allChunks: Chunk[] = [];

    for (let pIdx = 0; pIdx < parentGroups.length; pIdx++) {
        const parentText = parentGroups[pIdx]!;
        const parentId = randomUUID();
        const parentTokens = countTokens(parentText, tokenizer);

        const parentChunk: Chunk = {
            id: parentId,
            text: parentText,
            type: 'parent',
            metaData: {
                ...metaData,
                chunkIndex: pIdx,
            },
            tokens: parentTokens,
        };
        allChunks.push(parentChunk);

        const childUnits = extractChildUnits(parentText);

        const childChunksForParent: Chunk[] = [];
        let currentChild = '';

        function flushChild() {
            const trimmed = currentChild.trim();
            if (!trimmed) return;

            const childTokens = countTokens(trimmed, tokenizer);
            const childChunk: Chunk = {
                id: randomUUID(),
                text: trimmed,
                type: 'child',
                parentId,
                metaData: {
                    ...metaData,
                    parentId,
                    parentIndex: pIdx,
                    childIndex: childChunksForParent.length,
                },
                tokens: childTokens,
            };

            childChunksForParent.push(childChunk);
            allChunks.push(childChunk);
            currentChild = '';
        }

        for (const unit of childUnits) {
        if (unit.isList) {
            const candidate = currentChild ? `${currentChild}\n\n${unit.text}` : unit.text;

            if (countTokens(candidate, tokenizer) > maxChildTokens && currentChild) {
                flushChild();
            }

            const freshCandidate = currentChild ? `${currentChild}\n\n${unit.text}` : unit.text;
            if (countTokens(freshCandidate, tokenizer) <= maxChildTokens) {
                currentChild = freshCandidate;
            } else {
                const items = unit.text.split('\n');
                for (const item of items) {
                    const itemCand = currentChild ? `${currentChild}\n${item}` : item;
                    if (countTokens(itemCand, tokenizer) > maxChildTokens && currentChild) {
                        flushChild();
                        currentChild = item;
                    } else {
                        currentChild = itemCand;
                    }
                }
            }
        } else {
            const candidate = currentChild ? `${currentChild} ${unit.text}` : unit.text;
            if (countTokens(candidate, tokenizer) > maxChildTokens && currentChild) {
                flushChild();
                currentChild = unit.text;
            } else {
                currentChild = candidate;
            }
        }
        }
        flushChild();

        parentChunk.metaData.totalChildChunks = childChunksForParent.length;
    }

    return allChunks;
}

export function getHierarchicalView(chunks: Chunk[]): Array<{ parent: Chunk; children: Chunk[] }> {
    const parents = chunks.filter((c) => c.type === 'parent');
    const children = chunks.filter((c) => c.type === 'child');

    const childrenByParentId = new Map<string, Chunk[]>();
    for (const child of children) {
        if (child.parentId) {
            const list = childrenByParentId.get(child.parentId) || [];
            list.push(child);
            childrenByParentId.set(child.parentId, list);
        }
    }

    return parents.map((parent) => ({
        parent,
        children: childrenByParentId.get(parent.id) || [],
    }));
}