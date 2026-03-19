package proxy

import (
	"encoding/json"
	"testing"
)

func TestCleanIncompleteToolCalls_OpenAIOrphanedLast(t *testing.T) {
	// Last assistant message has tool_calls with no matching tool responses → remove them
	body := `{
		"messages": [
			{"role": "user", "content": "hello"},
			{"role": "assistant", "content": "let me check", "tool_calls": [
				{"id": "call_1", "type": "function", "function": {"name": "read_file", "arguments": "{}"}}
			]}
		]
	}`

	cleaned, err := cleanIncompleteToolCalls([]byte(body))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	var req map[string]interface{}
	if err := json.Unmarshal(cleaned, &req); err != nil {
		t.Fatalf("failed to unmarshal cleaned body: %v", err)
	}

	messages := req["messages"].([]interface{})
	if len(messages) != 2 {
		t.Fatalf("expected 2 messages, got %d", len(messages))
	}

	assistantMsg := messages[1].(map[string]interface{})
	if _, has := assistantMsg["tool_calls"]; has {
		t.Error("expected tool_calls to be removed from last assistant message")
	}
	if assistantMsg["content"] != "let me check" {
		t.Errorf("expected content preserved, got %v", assistantMsg["content"])
	}
}

func TestCleanIncompleteToolCalls_OpenAIOrphanedMid(t *testing.T) {
	// Mid-conversation orphaned tool_calls → insert placeholder tool responses
	body := `{
		"messages": [
			{"role": "user", "content": "hello"},
			{"role": "assistant", "content": "", "tool_calls": [
				{"id": "call_1", "type": "function", "function": {"name": "read_file", "arguments": "{}"}},
				{"id": "call_2", "type": "function", "function": {"name": "write_file", "arguments": "{}"}}
			]},
			{"role": "tool", "tool_call_id": "call_1", "content": "file content"},
			{"role": "user", "content": "now do more"},
			{"role": "assistant", "content": "", "tool_calls": [
				{"id": "call_3", "type": "function", "function": {"name": "list_dir", "arguments": "{}"}}
			]}
		]
	}`

	cleaned, err := cleanIncompleteToolCalls([]byte(body))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	var req map[string]interface{}
	if err := json.Unmarshal(cleaned, &req); err != nil {
		t.Fatalf("failed to unmarshal cleaned body: %v", err)
	}

	messages := req["messages"].([]interface{})

	// Should have inserted a placeholder for call_2 and removed call_3
	// Original: user, assistant(call_1,call_2), tool(call_1), user, assistant(call_3)
	// Expected: user, assistant(call_1,call_2), tool(call_1), tool(call_2-placeholder), user, assistant(no tool_calls)
	foundPlaceholder := false
	for _, m := range messages {
		msg := m.(map[string]interface{})
		role, _ := msg["role"].(string)
		if role == "tool" {
			callID, _ := msg["tool_call_id"].(string)
			if callID == "call_2" {
				foundPlaceholder = true
			}
		}
	}

	if !foundPlaceholder {
		t.Error("expected placeholder tool response for call_2")
	}

	// Last assistant should have tool_calls removed (call_3 was orphaned)
	lastMsg := messages[len(messages)-1].(map[string]interface{})
	if lastMsg["role"] != "assistant" {
		t.Fatalf("expected last message to be assistant, got %v", lastMsg["role"])
	}
	if _, has := lastMsg["tool_calls"]; has {
		t.Error("expected tool_calls to be removed from last assistant message")
	}
}

func TestCleanIncompleteToolCalls_OpenAINoOrphans(t *testing.T) {
	// All tool_calls have matching tool responses → no changes
	body := `{
		"messages": [
			{"role": "user", "content": "hello"},
			{"role": "assistant", "content": "", "tool_calls": [
				{"id": "call_1", "type": "function", "function": {"name": "read_file", "arguments": "{}"}}
			]},
			{"role": "tool", "tool_call_id": "call_1", "content": "ok"},
			{"role": "assistant", "content": "done"}
		]
	}`

	cleaned, err := cleanIncompleteToolCalls([]byte(body))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Should be unchanged (return original bytes)
	if string(cleaned) != body {
		t.Error("expected body to be unchanged when no orphans exist")
	}
}

func TestCleanIncompleteToolCalls_OpenAIPartialOrphaned(t *testing.T) {
	// Last assistant has 2 tool_calls, only 1 has a response → remove only the orphaned one
	body := `{
		"messages": [
			{"role": "user", "content": "hello"},
			{"role": "assistant", "content": "", "tool_calls": [
				{"id": "call_1", "type": "function", "function": {"name": "read_file", "arguments": "{}"}},
				{"id": "call_2", "type": "function", "function": {"name": "write_file", "arguments": "{}"}}
			]},
			{"role": "tool", "tool_call_id": "call_1", "content": "ok"}
		]
	}`

	cleaned, err := cleanIncompleteToolCalls([]byte(body))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	var req map[string]interface{}
	if err := json.Unmarshal(cleaned, &req); err != nil {
		t.Fatalf("failed to unmarshal: %v", err)
	}

	messages := req["messages"].([]interface{})
	assistantMsg := messages[1].(map[string]interface{})
	toolCalls, ok := assistantMsg["tool_calls"].([]interface{})
	if !ok {
		t.Fatal("expected tool_calls to still exist (with kept calls)")
	}
	if len(toolCalls) != 1 {
		t.Fatalf("expected 1 kept tool_call, got %d", len(toolCalls))
	}

	tc := toolCalls[0].(map[string]interface{})
	if tc["id"] != "call_1" {
		t.Errorf("expected kept tool_call to be call_1, got %v", tc["id"])
	}
}
