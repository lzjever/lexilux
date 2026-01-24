# Models.dev API JSON Format Report

**Generated:** 2025-01-27  
**Source:** https://models.dev/api.json  
**File Size:** 941,616 bytes (~920 KB)  
**File Type:** JSON text data

## Executive Summary

The Models.dev API provides a comprehensive database of AI model specifications, pricing, and features across 82 providers, containing information about 2,251 unique AI models. The data is structured as a JSON object where each key represents a provider, and each provider contains metadata and a dictionary of models.

## Overall Structure

### Root Level
- **Type:** Object (dictionary)
- **Structure:** Key-value pairs where:
  - **Key:** Provider identifier (string, e.g., "openai", "anthropic", "google")
  - **Value:** Provider object (dictionary)

### Provider Count
- **Total Providers:** 82
- **Provider Examples:** `privatemode-ai`, `moonshotai-cn`, `firmware`, `lucidquery`, `moonshotai`, `openai`, `anthropic`, `google`, `azure`, etc.

## Provider Object Structure

Each provider object contains the following fields:

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `id` | string | Unique provider identifier | `"openai"` |
| `name` | string | Human-readable provider name | `"OpenAI"` |
| `api` | string | Base API endpoint URL | `"https://api.openai.com/v1"` |
| `doc` | string | Documentation URL | `"https://platform.openai.com/docs"` |
| `npm` | string | NPM package name for SDK | `"@ai-sdk/openai"` |
| `env` | array[string] | Required environment variable names | `["OPENAI_API_KEY"]` |
| `models` | object | Dictionary of models (keyed by model ID) | See Models Structure below |

### Provider-Level Fields Details

- **`id`**: Always matches the key in the root object
- **`env`**: Array of environment variable names needed for authentication
- **`npm`**: NPM package identifier for the AI SDK
- **`api`**: Base URL for the provider's API endpoint
- **`doc`**: Link to provider documentation
- **`name`**: Display name for the provider

## Models Structure

The `models` field is a **dictionary** (not an array) where:
- **Key:** Model identifier (string, e.g., `"gpt-4"`, `"claude-3-opus"`)
- **Value:** Model specification object

**Important:** All 82 providers use a dictionary structure for models (no providers use arrays).

## Model Object Structure

Each model object contains the following fields:

### Required Fields (100% presence)

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Unique model identifier |
| `name` | string | Human-readable model name |
| `attachment` | boolean | Whether the model supports file attachments |
| `reasoning` | boolean | Whether the model supports reasoning/chain-of-thought |
| `tool_call` | boolean | Whether the model supports function/tool calling |
| `last_updated` | string | Date when model info was last updated (YYYY-MM-DD format) |
| `modalities` | object | Input/output modalities (see below) |
| `open_weights` | boolean | Whether the model has open weights |
| `limit` | object | Context and output limits (see below) |

### Common Fields (>90% presence)

| Field | Type | Presence | Description |
|-------|------|----------|-------------|
| `temperature` | boolean | 98.3% | Whether the model supports temperature parameter |
| `cost` | object | 95.9% | Pricing information (see below) |
| `family` | string | 92.4% | Model family name (e.g., "gpt", "claude", "llama") |
| `release_date` | string | 100% | Model release date (YYYY-MM-DD or YYYY-MM format) |

### Optional Fields

| Field | Type | Presence | Description |
|-------|------|----------|-------------|
| `knowledge` | string | 65.9% | Knowledge cutoff date (YYYY-MM format) |
| `structured_output` | boolean | 21.9% | Whether the model supports structured outputs |
| `interleaved` | boolean/dict | 4.2% | Interleaved content support (can be boolean or object with `field` property) |
| `status` | string | 1.4% | Model status (e.g., "deprecated") |
| `provider` | object | 1.6% | Provider-specific metadata (contains `npm` field) |

## Nested Object Structures

### `modalities` Object
```json
{
  "input": ["text", "image", "audio"],
  "output": ["text", "json"]
}
```
- **`input`**: Array of supported input types (strings: "text", "image", "audio", etc.)
- **`output`**: Array of supported output types (strings: "text", "json", etc.)

### `limit` Object
```json
{
  "context": 128000,
  "output": 8192
}
```
- **`context`**: Maximum context window size in tokens (integer, 0 means unlimited/unknown)
- **`output`**: Maximum output length in tokens (integer)

### `cost` Object
```json
{
  "input": 0.00001,
  "output": 0.00003
}
```
- **`input`**: Cost per token for input (number, typically in USD)
- **`output`**: Cost per token for output (number, typically in USD)
- Note: Value of `0` may indicate free tier or pricing not available

### `interleaved` Object (when present as object)
```json
{
  "field": "reasoning_content"
}
```
- **`field`**: Field name for interleaved content

## Statistics

### Model Distribution

- **Total Models:** 2,251
- **Average Models per Provider:** ~27.5
- **Largest Provider:** Vercel (184 models)
- **Smallest Providers:** Xiaomi, Vivgrid, Kimi-for-coding (1 model each)

### Top 10 Providers by Model Count

1. Vercel: 184 models
2. OpenRouter: 140 models
3. Poe: 115 models
4. Azure: 93 models
5. Helicone: 91 models
6. Azure Cognitive Services: 91 models
7. Novita AI: 77 models
8. SiliconFlow: 73 models
9. Cloudflare Workers AI: 73 models
10. SiliconFlow CN: 71 models

### Model Families

The database contains models from various families. Top 15 families:

1. **Qwen**: 343 models
2. **Llama**: 156 models
3. **GPT**: 123 models
4. **GLM**: 97 models
5. **DeepSeek**: 92 models
6. **Grok**: 83 models
7. **Claude Sonnet**: 70 models
8. **Claude Opus**: 59 models
9. **GPT Codex**: 57 models
10. **Kimi**: 51 models
11. **GPT-OSS**: 50 models
12. **Gemini Flash**: 49 models
13. **Phi**: 45 models
14. **DeepSeek Thinking**: 41 models
15. **Claude Haiku**: 40 models

## Field Type Analysis

### Boolean Fields
- `attachment`, `reasoning`, `tool_call`, `open_weights`, `temperature`, `structured_output`
- All boolean fields use `true`/`false` values

### String Fields
- `id`, `name`, `family`, `knowledge`, `release_date`, `last_updated`, `status`
- Date formats: `YYYY-MM-DD` or `YYYY-MM`
- Knowledge cutoff: `YYYY-MM` format

### Numeric Fields
- `cost.input`, `cost.output`: Numbers (floats)
- `limit.context`, `limit.output`: Integers

### Object Fields
- `modalities`, `limit`, `cost`, `interleaved` (when object), `provider`

### Array Fields
- `modalities.input`, `modalities.output`: Arrays of strings
- `env`: Array of strings (at provider level)

## Data Quality Observations

1. **Consistency:** All providers follow the same structure (models as dictionary)
2. **Completeness:** Core fields (id, name, attachment, reasoning, etc.) are present in 100% of models
3. **Optional Fields:** Some fields like `knowledge` (65.9%) and `structured_output` (21.9%) are optional
4. **Date Formats:** Consistent date formatting (YYYY-MM-DD or YYYY-MM)
5. **Cost Data:** 95.9% of models have cost information

## Example Provider Entry

```json
{
  "id": "privatemode-ai",
  "env": ["PRIVATEMODE_API_KEY", "PRIVATEMODE_ENDPOINT"],
  "npm": "@ai-sdk/openai-compatible",
  "api": "http://localhost:8080/v1",
  "name": "Privatemode AI",
  "doc": "https://docs.privatemode.ai/api/overview",
  "models": {
    "whisper-large-v3": {
      "id": "whisper-large-v3",
      "name": "Whisper large-v3",
      "family": "whisper",
      "attachment": true,
      "reasoning": false,
      "tool_call": false,
      "structured_output": false,
      "temperature": true,
      "knowledge": "2023-09",
      "release_date": "2023-09-01",
      "last_updated": "2023-09-01",
      "modalities": {
        "input": ["audio"],
        "output": ["text"]
      },
      "open_weights": true,
      "cost": {
        "input": 0,
        "output": 0
      },
      "limit": {
        "context": 0,
        "output": 4096
      }
    }
  }
}
```

## Use Cases

This API format is suitable for:

1. **Model Discovery:** Finding available models across providers
2. **Feature Comparison:** Comparing capabilities (reasoning, tool calling, modalities)
3. **Pricing Analysis:** Comparing costs across providers
4. **SDK Integration:** Using `npm` field for SDK selection
5. **API Integration:** Using `api` field for endpoint configuration
6. **Capability Filtering:** Filtering models by features (attachment, reasoning, etc.)

## Technical Notes

- **File Format:** Valid JSON (UTF-8 encoded)
- **Size:** ~941 KB uncompressed
- **Structure:** Single-level provider keys, nested model dictionaries
- **No Arrays:** Models are always stored as dictionaries (keyed by model ID)
- **Consistency:** Uniform structure across all providers

## Conclusion

The Models.dev API provides a well-structured, comprehensive database of AI model information. The format is consistent, well-documented through field presence, and suitable for programmatic access. The data covers 82 providers and 2,251 models with detailed specifications including capabilities, pricing, limits, and metadata.
