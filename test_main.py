from main import split_blocks, chunk_text_by_chars

def test_split_blocks():
    # Test simple split
    text = "Block 1\n\nBlock 2"
    assert split_blocks(text) == ["Block 1", "Block 2"]

    # Test split with spaces
    text = "Block 1\n  \nBlock 2"
    assert split_blocks(text) == ["Block 1", "Block 2"]

    # Test multiple newlines
    text = "Block 1\n\n\n\nBlock 2"
    assert split_blocks(text) == ["Block 1", "Block 2"]

    # Test no split (single newline)
    text = "Block 1\nBlock 2"
    assert split_blocks(text) == ["Block 1\nBlock 2"]

    # Test empty input
    assert split_blocks("") == []
    assert split_blocks("\n\n") == []

def test_chunk_text_by_chars_empty():
    assert chunk_text_by_chars("", 100) == []

def test_chunk_text_by_chars_single_chunk():
    text = "Block 1\n\nBlock 2\n\nBlock 3"
    # target_tokens=10 => target_chars=40. Each block is 7 chars.
    # 7 + 7 + 7 = 21 < 40. Should all be in one chunk.
    chunks = chunk_text_by_chars(text, 10)
    assert len(chunks) == 1
    assert chunks[0] == "Block 1\n\nBlock 2\n\nBlock 3"

def test_chunk_text_by_chars_multi_chunk():
    text = "Block 1\n\nBlock 2\n\nBlock 3"
    # target_tokens=3 => target_chars=12.
    # Block 1 (7 chars) -> current=[B1], len=7
    # Block 2 (7 chars): 7+7=14 > 12. Flush B1. current=[B2], len=7
    # Block 3 (7 chars): 7+7=14 > 12. Flush B2. current=[B3], len=7
    # End: Flush B3.
    chunks = chunk_text_by_chars(text, 3)
    assert len(chunks) == 3
    assert chunks == ["Block 1", "Block 2", "Block 3"]

def test_chunk_text_by_chars_oversized_block():
    text = "A very long block that exceeds the limit by itself"
    # target_tokens=2 => target_chars=8.
    # Block is 50 chars.
    # Since it's the first block, it's added to current.
    # Loop ends, current is flushed.
    chunks = chunk_text_by_chars(text, 2)
    assert len(chunks) == 1
    assert chunks[0] == text

def test_chunk_text_by_chars_boundary():
    # Test when current_len + blk_len is EXACTLY target_chars
    text = "1234\n\n5678"
    # target_tokens=2 => target_chars=8.
    # Block 1: "1234" (4 chars). current_len=4.
    # Block 2: "5678" (4 chars). 4+4=8. NOT > 8.
    # So "5678" is added to current.
    # Result should be one chunk "1234\n\n5678"
    chunks = chunk_text_by_chars(text, 2)
    assert len(chunks) == 1
    assert chunks[0] == "1234\n\n5678"

    # Test when current_len + blk_len is target_chars + 1
    text = "1234\n\n56789"
    # target_tokens=2 => target_chars=8.
    # Block 1: "1234" (4 chars). current_len=4.
    # Block 2: "56789" (5 chars). 4+5=9 > 8. Flush "1234".
    # Result: ["1234", "56789"]
    chunks = chunk_text_by_chars(text, 2)
    assert len(chunks) == 2
    assert chunks == ["1234", "56789"]
