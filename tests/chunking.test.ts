import { describe, expect, it } from 'bun:test';
import {
  BGE_MODEL_LIMIT,
  DEFAULT_MAX_CHILD_TOKENS,
  DEFAULT_MAX_PARENT_TOKENS,
  chunkDoc,
  cleanPdfText,
  countTokensWithBge,
  getBgeTokenizer,
  getHierarchicalView,
  isHeading,
  isListItem,
} from '../src/chunking.ts';

describe('chunking pipeline with bge-small & simplified text cleaning', () => {
  describe('cleanPdfText - line breaks and bullet handling', () => {
    it('unwraps single line breaks inside a paragraph into a single continuous block', () => {
      const raw = `This is a paragraph extracted
from a PDF where lines are wrapped
at fixed width boundaries.`;

      const cleaned = cleanPdfText(raw);
      expect(cleaned).toBe(
        'This is a paragraph extracted from a PDF where lines are wrapped at fixed width boundaries.'
      );
    });

    it('preserves double line breaks as separate paragraphs', () => {
      const raw = `Paragraph one text here.

Paragraph two text here.


Paragraph three text here.`;

      const cleaned = cleanPdfText(raw);
      expect(cleaned).toBe('Paragraph one text here.\n\nParagraph two text here.\n\nParagraph three text here.');
    });

    it('identifies bullet points and keeps them on their own lines', () => {
      const raw = `Here are the specifications:
• First feature: high performance
• Second feature: robust error handling
• Third feature: hierarchical chunking`;

      const cleaned = cleanPdfText(raw);
      expect(cleaned).toBe(
        'Here are the specifications:\n• First feature: high performance\n• Second feature: robust error handling\n• Third feature: hierarchical chunking'
      );
    });

    it('joins soft-wrapped continuation lines within a bullet point to that bullet item', () => {
      const raw = `• This is a long bullet item that wraps
  onto the next line before concluding.
• This is the second bullet item.`;

      const cleaned = cleanPdfText(raw);
      expect(cleaned).toBe(
        '• This is a long bullet item that wraps onto the next line before concluding.\n• This is the second bullet item.'
      );
    });

    it('handles various bullet styles: unicode, dashes, asterisks, numbered lists', () => {
      const raw = `* Asterisk bullet
- Dash bullet
+ Plus bullet
▪ Square bullet
1. Numbered item 1
2) Numbered item 2
(a) Alphabetic item a`;

      const cleaned = cleanPdfText(raw);
      const lines = cleaned.split('\n');
      expect(lines).toHaveLength(7);
      expect(lines.every((l) => isListItem(l))).toBe(true);
    });

    it('resolves line-end hyphenation across wrapped lines', () => {
      const raw = `This sentence contains a hierar-
chical structure and multi-
faceted elements.`;

      const cleaned = cleanPdfText(raw);
      expect(cleaned).toBe(
        'This sentence contains a hierarchical structure and multifaceted elements.'
      );
    });

    it('strips markdown images and tables', () => {
      const raw = `Introduction text.

![sample image](https://example.com/img.png)

| Header 1 | Header 2 |
| --- | --- |
| Row 1 | Row 2 |

Conclusion text.`;

      const cleaned = cleanPdfText(raw);
      expect(cleaned).toBe('Introduction text.\n\nConclusion text.');
    });
  });

  describe('bge-small tokenizer integration', () => {
    it('initializes bge-small tokenizer and counts tokens accurately', async () => {
      const tokenizer = await getBgeTokenizer();
      expect(tokenizer).toBeDefined();

      const count = await countTokensWithBge('Hello world! Testing BGE-small token counting.');
      expect(count).toBeGreaterThan(0);
      expect(typeof count).toBe('number');
    });

    it('verifies BGE model limits and default constants', () => {
      expect(BGE_MODEL_LIMIT).toBe(512);
      expect(DEFAULT_MAX_CHILD_TOKENS).toBe(400);
      expect(DEFAULT_MAX_PARENT_TOKENS).toBe(1000);
      expect(DEFAULT_MAX_CHILD_TOKENS).toBeLessThanOrEqual(BGE_MODEL_LIMIT);
    });
  });

  describe('isListItem and isHeading', () => {
    it('accurately identifies list items', () => {
      expect(isListItem('• Item')).toBe(true);
      expect(isListItem('- Item')).toBe(true);
      expect(isListItem('* Item')).toBe(true);
      expect(isListItem('1. Item')).toBe(true);
      expect(isListItem('(1) Item')).toBe(true);
      expect(isListItem('a) Item')).toBe(true);
      expect(isListItem('Normal text.')).toBe(false);
      expect(isListItem('# Heading')).toBe(false);
    });

    it('accurately identifies headings', () => {
      expect(isHeading('# Title')).toBe(true);
      expect(isHeading('## Subtitle')).toBe(true);
      expect(isHeading('### Section')).toBe(true);
      expect(isHeading('Paragraph text')).toBe(false);
      expect(isHeading('• Bullet')).toBe(false);
    });
  });

  describe('chunkDoc - Child list atomicity & Hierarchical Chunking', () => {
    it('only adds child lists to a chunk if ALL of it can be included, otherwise makes a different chunk', async () => {
      // Create a scenario where a paragraph takes some tokens, and a list cannot fit together with it
      const text = `This introductory paragraph has enough text to fill a substantial portion of the child token budget.

Features list:
• Feature item one with descriptive details
• Feature item two with technical specs
• Feature item three with integration notes`;

      // Set a token limit where paragraph + list > limit, but list on its own <= limit
      const chunks = await chunkDoc(text, {}, { maxParentTokens: 1000, maxChildTokens: 35 });
      const children = chunks.filter((c) => c.type === 'child');

      // The list should NOT be broken across chunks. The entire list should be in its own chunk!
      const listChunk = children.find((c) => c.text.includes('• Feature item one'));
      expect(listChunk).toBeDefined();
      expect(listChunk?.text).toContain('• Feature item one');
      expect(listChunk?.text).toContain('• Feature item two');
      expect(listChunk?.text).toContain('• Feature item three');
    });

    it('creates parent and child chunks with proper parentId linkage and respects 400 child / 1000 parent tokens', async () => {
      const sample = `
This is the first paragraph. It describes the overview of the architecture and data flow.
Single line breaks inside paragraphs
are handled cleanly.

Here are the primary components:
• Ingestion module: processes raw PDF bytes and extracts text.
• Normalizer module: cleans line wraps and isolates bullet lists.
• Hierarchical chunker: creates parent and child chunks.

The conclusion summarizes the benefits of this hierarchical approach.
Retrieval accuracy is improved because vector search targets child chunks,
while LLM context uses parent chunks.
`;

      const chunks = await chunkDoc(
        sample,
        { source: 'unit-test.pdf' },
        { maxParentTokens: 1000, maxChildTokens: 400 }
      );

      const parents = chunks.filter((c) => c.type === 'parent');
      const children = chunks.filter((c) => c.type === 'child');

      expect(parents.length).toBeGreaterThanOrEqual(1);
      expect(children.length).toBeGreaterThanOrEqual(parents.length);

      // Verify every child has a valid parentId pointing to an existing parent
      const parentIds = new Set(parents.map((p) => p.id));
      for (const child of children) {
        expect(child.parentId).toBeDefined();
        expect(parentIds.has(child.parentId!)).toBe(true);
        expect(child.metaData.source).toBe('unit-test.pdf');
        expect(child.tokens).toBeLessThanOrEqual(400);
        expect(child.tokens).toBeLessThanOrEqual(BGE_MODEL_LIMIT);
      }

      // Verify parent metadata contains totalChildChunks
      for (const parent of parents) {
        expect(parent.metaData.totalChildChunks).toBeGreaterThan(0);
        expect(parent.tokens).toBeLessThanOrEqual(1000);
        const childCountForThisParent = children.filter((c) => c.parentId === parent.id).length;
        expect(parent.metaData.totalChildChunks).toBe(childCountForThisParent);
      }
    });

    it('clamps child tokens if requested value exceeds bge-small 512 limit', async () => {
      const sample = 'Word '.repeat(300).trim();
      const chunks = await chunkDoc(sample, {}, { maxChildTokens: 600 });
      const children = chunks.filter((c) => c.type === 'child');

      for (const child of children) {
        expect(child.tokens).toBeLessThanOrEqual(BGE_MODEL_LIMIT);
      }
    });

    it('returns empty array when given empty input', async () => {
      expect(await chunkDoc('')).toEqual([]);
      expect(await chunkDoc('   \n\n   ')).toEqual([]);
    });
  });

  describe('getHierarchicalView', () => {
    it('groups child chunks under their corresponding parent chunk', async () => {
      const sample = `
Section one paragraph is here with some text.

• Bullet A
• Bullet B

Section two paragraph with more text.
`;

      const chunks = await chunkDoc(sample, {}, { maxParentTokens: 1000, maxChildTokens: 400 });
      const view = getHierarchicalView(chunks);

      expect(view.length).toBeGreaterThan(0);
      for (const group of view) {
        expect(group.parent.type).toBe('parent');
        for (const child of group.children) {
          expect(child.type).toBe('child');
          expect(child.parentId).toBe(group.parent.id);
        }
      }
    });
  });
});
