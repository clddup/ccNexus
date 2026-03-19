package proxy

import (
	"encoding/json"
	"net/http"
	"strings"

	"github.com/lich0821/ccNexus/internal/logger"
	"github.com/lich0821/ccNexus/internal/tokencount"
)

// normalizeAPIUrl ensures the API URL has a protocol prefix
func normalizeAPIUrl(apiUrl string) string {
	if !strings.HasPrefix(apiUrl, "http://") && !strings.HasPrefix(apiUrl, "https://") {
		return "https://" + apiUrl
	}
	return apiUrl
}

// shouldRetry determines if a response should trigger a retry
func shouldRetry(statusCode int) bool {
	return statusCode != http.StatusOK &&
		statusCode != http.StatusBadRequest &&
		statusCode != http.StatusUnauthorized
}

// cleanIncompleteToolCalls removes incomplete tool_use blocks and fixes orphaned
// OpenAI-format tool_calls (assistant messages with tool_calls that have no
// matching tool response messages) from request.
func cleanIncompleteToolCalls(bodyBytes []byte) ([]byte, error) {
	var req map[string]interface{}
	if err := json.Unmarshal(bodyBytes, &req); err != nil {
		return bodyBytes, err
	}

	messages, ok := req["messages"].([]interface{})
	if !ok {
		return bodyBytes, nil
	}

	modified := false

	// Pass 1: Clean incomplete Claude-format tool_use blocks (no input)
	for i := len(messages) - 1; i >= 0; i-- {
		msg, ok := messages[i].(map[string]interface{})
		if !ok {
			continue
		}

		role, _ := msg["role"].(string)
		if role != "assistant" {
			break
		}

		content, ok := msg["content"].([]interface{})
		if !ok {
			break
		}

		var cleanedContent []interface{}
		hasIncomplete := false
		for _, block := range content {
			blockMap, ok := block.(map[string]interface{})
			if !ok {
				cleanedContent = append(cleanedContent, block)
				continue
			}

			blockType, _ := blockMap["type"].(string)
			if blockType == "tool_use" {
				if input, hasInput := blockMap["input"]; !hasInput || input == nil {
					logger.Debug("Removing incomplete tool_use block without input")
					hasIncomplete = true
					continue
				}
			}
			cleanedContent = append(cleanedContent, block)
		}

		if hasIncomplete {
			modified = true
			if len(cleanedContent) == 0 {
				messages = append(messages[:i], messages[i+1:]...)
			} else {
				msg["content"] = cleanedContent
			}
		}
		break
	}

	// Pass 2: Fix orphaned OpenAI-format tool_calls
	// Collect all tool_call_ids that have a matching "tool" role response
	respondedIDs := make(map[string]bool)
	for _, m := range messages {
		msg, ok := m.(map[string]interface{})
		if !ok {
			continue
		}
		role, _ := msg["role"].(string)
		if role == "tool" {
			if callID, ok := msg["tool_call_id"].(string); ok && callID != "" {
				respondedIDs[callID] = true
			}
		}
	}

	// Find assistant messages with tool_calls that have orphaned IDs
	var newMessages []interface{}
	for i, m := range messages {
		msg, ok := m.(map[string]interface{})
		if !ok {
			newMessages = append(newMessages, m)
			continue
		}

		role, _ := msg["role"].(string)
		if role != "assistant" {
			newMessages = append(newMessages, m)
			continue
		}

		toolCalls, hasTCs := msg["tool_calls"].([]interface{})
		if !hasTCs || len(toolCalls) == 0 {
			newMessages = append(newMessages, m)
			continue
		}

		// Check which tool_call IDs are orphaned (no matching tool response)
		var orphanedIDs []string
		var keptToolCalls []interface{}
		for _, tc := range toolCalls {
			tcMap, ok := tc.(map[string]interface{})
			if !ok {
				keptToolCalls = append(keptToolCalls, tc)
				continue
			}
			callID, _ := tcMap["id"].(string)
			if callID == "" || respondedIDs[callID] {
				keptToolCalls = append(keptToolCalls, tc)
			} else {
				orphanedIDs = append(orphanedIDs, callID)
			}
		}

		if len(orphanedIDs) == 0 {
			newMessages = append(newMessages, m)
			continue
		}

		modified = true

		// Check if this is the last assistant message with tool_calls
		isLast := true
		for j := i + 1; j < len(messages); j++ {
			laterMsg, ok := messages[j].(map[string]interface{})
			if !ok {
				continue
			}
			if laterRole, _ := laterMsg["role"].(string); laterRole == "assistant" {
				if _, has := laterMsg["tool_calls"]; has {
					isLast = false
					break
				}
			}
		}

		if isLast {
			// Last assistant message: remove orphaned tool_calls
			logger.Debug("Removing %d orphaned tool_calls from last assistant message", len(orphanedIDs))
			if len(keptToolCalls) > 0 {
				msg["tool_calls"] = keptToolCalls
			} else {
				delete(msg, "tool_calls")
				// Ensure content is not empty for the message to be valid
				if msg["content"] == nil || msg["content"] == "" {
					msg["content"] = ""
				}
			}
			newMessages = append(newMessages, msg)
		} else {
			// Mid-conversation: keep all tool_calls but insert placeholder tool responses
			newMessages = append(newMessages, msg)
			for _, orphanID := range orphanedIDs {
				logger.Debug("Inserting placeholder tool response for orphaned tool_call_id: %s", orphanID)
				newMessages = append(newMessages, map[string]interface{}{
					"role":         "tool",
					"tool_call_id": orphanID,
					"content":      "",
				})
			}
		}
	}

	if !modified {
		return bodyBytes, nil
	}

	req["messages"] = newMessages
	return json.Marshal(req)
}

// estimateInputTokens estimates input tokens from request body
func (p *Proxy) estimateInputTokens(bodyBytes []byte) int {
	var req tokencount.CountTokensRequest
	if json.Unmarshal(bodyBytes, &req) == nil {
		return tokencount.EstimateInputTokens(&req)
	}
	return 0
}

// estimateTokens estimates tokens when API doesn't provide usage
func (p *Proxy) estimateTokens(bodyBytes []byte, outputText string, inputTokens, outputTokens int, endpointName string) (int, int) {
	if inputTokens == 0 {
		var req tokencount.CountTokensRequest
		if json.Unmarshal(bodyBytes, &req) == nil {
			inputTokens = tokencount.EstimateInputTokens(&req)
			logger.Debug("[%s] Estimated input tokens: %d", endpointName, inputTokens)
		}
	}

	if outputTokens == 0 && outputText != "" {
		outputTokens = tokencount.EstimateOutputTokens(outputText)
		logger.Debug("[%s] Estimated output tokens: %d", endpointName, outputTokens)
	}

	return inputTokens, outputTokens
}
