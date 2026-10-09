/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.flink.agents.api.agents;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.apache.flink.agents.api.prompt.Prompt;
import org.apache.flink.agents.api.resource.PythonResourceDescriptor;
import org.apache.flink.agents.api.resource.ResourceDescriptor;
import org.apache.flink.agents.api.resource.ResourceType;
import org.apache.flink.agents.api.skills.SkillSourceSpec;
import org.apache.flink.agents.api.skills.Skills;
import org.apache.flink.api.common.typeinfo.BasicTypeInfo;
import org.apache.flink.api.common.typeinfo.TypeInformation;
import org.apache.flink.api.java.typeutils.RowTypeInfo;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import javax.annotation.Nullable;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Function;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

public class ReActAgentTest {
    @Test
    public void testOutputSchemaSerialization() throws JsonProcessingException {
        ObjectMapper mapper = new ObjectMapper();
        RowTypeInfo typeInfo =
                new RowTypeInfo(
                        new TypeInformation[] {
                            BasicTypeInfo.INT_TYPE_INFO, BasicTypeInfo.STRING_TYPE_INFO
                        },
                        new String[] {"a", "b"});
        OutputSchema schema = new OutputSchema(typeInfo);
        String json = mapper.writeValueAsString(schema);
        OutputSchema deserialized = mapper.readValue(json, OutputSchema.class);
        Assertions.assertEquals(typeInfo, deserialized.getSchema());
    }

    @Test
    @DisplayName("An agent built on a schema Jackson cannot render reports it with the cause kept")
    public void testAgentRejectsSchemaThatCannotRender() {
        assertThatThrownBy(() -> agentWithSchema(FieldLess.class))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("FieldLess")
                .hasMessageContaining("cannot be rendered as a JSON Schema")
                .hasCauseInstanceOf(IllegalArgumentException.class);
    }

    @Test
    @DisplayName("An agent built on a self-referential schema reports the self-reference")
    public void testAgentRejectsSelfReferentialSchema() {
        assertThatThrownBy(() -> agentWithSchema(SelfReferential.class))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("SelfReferential")
                .hasMessageContaining("self-referential")
                .hasCauseInstanceOf(StackOverflowError.class);
    }

    @Test
    @DisplayName("An agent built on a member that renders to no properties still prompts with it")
    public void testAgentAcceptsSchemaWithFieldLessMember() {
        assertThat(schemaPromptOf(agentWithSchema(WithCallback.class)))
                .contains("\"count\":{\"type\":\"integer\"}")
                .contains("\"callback\":{\"type\":\"object\",\"properties\":{}}");
    }

    @Test
    @DisplayName("An agent built on a renderable schema prompts with its rendered JSON Schema")
    public void testAgentAcceptsRenderableSchema() {
        assertThat(schemaPromptOf(agentWithSchema(WithCount.class)))
                .contains(
                        "{\"type\":\"object\",\"properties\":{\"count\":{\"type\":\"integer\"}}}");
    }

    @Test
    @DisplayName("An output schema of neither supported kind reports the type it received")
    public void testUnsupportedOutputSchemaTypeReportsTheType() {
        assertThatThrownBy(() -> agentWithSchema("not-a-schema"))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("java.lang.String")
                .hasMessageContaining("must be a RowTypeInfo or a Pojo class");
    }

    @Test
    @DisplayName("Constructor skills alone become the chat model's skills")
    public void testConstructorOnlySkills() {
        ResourceDescriptor chatModel = descriptor(Map.of("model", "qwen3:8b"));

        ResourceDescriptor actual = chatModelOf(agentWithSkills(chatModel, List.of("a", "b")));

        assertThat(actual.<List<String>>getArgument("skills")).containsExactly("a", "b");
        assertThat(actual.<String>getArgument("model")).isEqualTo("qwen3:8b");
    }

    @Test
    @DisplayName("Without constructor skills the descriptor is used exactly as given")
    public void testDescriptorOnlySkills() {
        ResourceDescriptor chatModel = descriptor(Map.of("skills", List.of("a")));

        assertThat(chatModelOf(agentWithSkills(chatModel, null))).isSameAs(chatModel);
        assertThat(chatModelOf(new ReActAgent(chatModel, null, null))).isSameAs(chatModel);
    }

    @Test
    @DisplayName("An empty skills list adds nothing, so skills stay disabled")
    public void testEmptyConstructorSkillsAddNothing() {
        ResourceDescriptor chatModel = descriptor(Map.of("model", "qwen3:8b"));

        ResourceDescriptor actual = chatModelOf(agentWithSkills(chatModel, List.of()));

        assertThat(actual).isSameAs(chatModel);
        assertThat(actual.getInitialArguments()).doesNotContainKey("skills");
    }

    @Test
    @DisplayName("Descriptor skills come first, then constructor skills, each name kept once")
    public void testCombinedSkillsAreOrderedAndDeduplicated() {
        ResourceDescriptor chatModel = descriptor(Map.of("skills", List.of("a", "b")));

        ResourceDescriptor actual =
                chatModelOf(agentWithSkills(chatModel, List.of("b", "c", "a", "c")));

        assertThat(actual.<List<String>>getArgument("skills")).containsExactly("a", "b", "c");
        assertThat(chatModel.<List<String>>getArgument("skills")).containsExactly("a", "b");
    }

    @Test
    @DisplayName("Agents sharing one descriptor each get only their own constructor skills")
    public void testDescriptorSharedAcrossAgentsStaysIndependent() {
        // Writable collections, as a caller typically builds them, so a merge done in place would
        // show up as one agent's skills leaking into the other instead of an immutable-write error.
        ResourceDescriptor chatModel =
                new ResourceDescriptor(
                        "com.example.ChatModel",
                        new HashMap<>(Map.of("skills", new ArrayList<>(List.of("shared")))));

        ResourceDescriptor first = chatModelOf(agentWithSkills(chatModel, List.of("a")));
        ResourceDescriptor second = chatModelOf(agentWithSkills(chatModel, List.of("b")));

        assertThat(first.<List<String>>getArgument("skills")).containsExactly("shared", "a");
        assertThat(second.<List<String>>getArgument("skills")).containsExactly("shared", "b");
        assertThat(chatModel.<List<String>>getArgument("skills")).containsExactly("shared");
    }

    @Test
    @DisplayName("Constructor skills leave the descriptor's command policy untouched")
    public void testConstructorSkillsKeepAllowedCommands() {
        ResourceDescriptor chatModel =
                descriptor(
                        Map.of(
                                "skills", List.of("a"),
                                "allowed_commands", List.of("echo", "bc"),
                                "allowed_script_dirs", List.of("/opt/scripts")));

        ResourceDescriptor actual = chatModelOf(agentWithSkills(chatModel, List.of("b")));

        assertThat(actual.<List<String>>getArgument("allowed_commands"))
                .containsExactly("echo", "bc");
        assertThat(actual.<List<String>>getArgument("allowed_script_dirs"))
                .containsExactly("/opt/scripts");
    }

    @Test
    @DisplayName("Constructor skills on a Python chat model keep it a Python declaration")
    public void testConstructorSkillsKeepDescriptorLanguage() {
        ResourceDescriptor chatModel =
                PythonResourceDescriptor.Builder.newBuilder("my_module.MyChatModel")
                        .addInitialArgument("skills", List.of("a"))
                        .build();

        ResourceDescriptor actual = chatModelOf(agentWithSkills(chatModel, List.of("b")));

        assertThat(actual)
                .isInstanceOf(PythonResourceDescriptor.class)
                .extracting(ResourceDescriptor::getModule, ResourceDescriptor::getClazz)
                .containsExactly("my_module", "MyChatModel");
        assertThat(actual.<List<String>>getArgument("skills")).containsExactly("a", "b");
    }

    private static ReActAgent agentWithSchema(Object outputSchema) {
        return new ReActAgent(
                ResourceDescriptor.Builder.newBuilder("com.example.ChatModel").build(),
                null,
                outputSchema);
    }

    private static ReActAgent agentWithSkills(
            ResourceDescriptor chatModel, @Nullable List<String> skills) {
        return new ReActAgent(chatModel, null, null, skills);
    }

    /** An immutable argument map, so any write to the caller's descriptor fails the test. */
    private static ResourceDescriptor descriptor(Map<String, Object> arguments) {
        return new ResourceDescriptor("com.example.ChatModel", arguments);
    }

    private static ResourceDescriptor chatModelOf(ReActAgent agent) {
        return (ResourceDescriptor)
                agent.getResources().get(ResourceType.CHAT_MODEL).get("_default_chat_model");
    }

    private static String schemaPromptOf(ReActAgent agent) {
        Prompt schemaPrompt =
                (Prompt)
                        agent.getResources().get(ResourceType.PROMPT).get("_default_schema_prompt");
        return schemaPrompt.formatString(Map.of());
    }

    @Test
    @DisplayName("Skill source specs are registered as a marker on the chat model descriptor")
    public void testSkillsObjectRegistersSourcesMarker() {
        ResourceDescriptor chatModel = descriptor(Map.of("model", "qwen3:8b"));
        Skills skills = Skills.fromLocalDir("/tmp/skill-a", "/tmp/skill-b");

        ReActAgent agent = new ReActAgent(chatModel, null, null, skills);
        ResourceDescriptor actual = chatModelOf(agent);

        @SuppressWarnings("unchecked")
        List<SkillSourceSpec> sources = actual.getArgument("skills_sources");
        assertThat(sources).isNotNull();
        assertThat(sources).hasSize(2);
        assertThat(sources.get(0).getScheme()).isEqualTo("local");
        assertThat(sources.get(0).getParams()).containsEntry("path", "/tmp/skill-a");
        // The original descriptor is not modified.
        assertThat((Object) chatModel.getArgument("skills_sources")).isNull();
        // The Skills object is registered as a resource.
        assertThat(agent.getResources().get(ResourceType.SKILLS))
                .containsKey("_react_agent_skills");
    }

    @Test
    @DisplayName("Null Skills object adds nothing")
    public void testNullSkillsObjectIsNoOp() {
        ResourceDescriptor chatModel = descriptor(Map.of("model", "qwen3:8b"));

        ReActAgent agent = new ReActAgent(chatModel, null, null, (Skills) null);
        ResourceDescriptor actual = chatModelOf(agent);

        assertThat(actual).isSameAs(chatModel);
        assertThat((Object) actual.getArgument("skills_sources")).isNull();
        assertThat((Object) actual.getArgument("skills")).isNull();
    }

    @Test
    @DisplayName("Empty Skills object adds nothing")
    public void testEmptySkillsObjectIsNoOp() {
        ResourceDescriptor chatModel = descriptor(Map.of("model", "qwen3:8b"));
        Skills empty = new Skills(List.of());

        ReActAgent agent = new ReActAgent(chatModel, null, null, empty);
        ResourceDescriptor actual = chatModelOf(agent);

        assertThat(actual).isSameAs(chatModel);
        assertThat((Object) actual.getArgument("skills_sources")).isNull();
    }

    @Test
    @DisplayName("Skills object and descriptor skills coexist — merge happens at runtime")
    public void testSkillsObjectAndDescriptorSkillsCoexist() {
        ResourceDescriptor chatModel =
                descriptor(Map.of("skills", List.of("a", "b"), "model", "qwen3:8b"));
        Skills skills = Skills.fromLocalDir("/tmp/skill-c");

        ReActAgent agent = new ReActAgent(chatModel, null, null, skills);
        ResourceDescriptor actual = chatModelOf(agent);

        // Explicit skills are preserved on the descriptor.
        assertThat(actual.<List<String>>getArgument("skills")).containsExactly("a", "b");
        // The skills_sources marker is present for runtime resolution.
        @SuppressWarnings("unchecked")
        List<SkillSourceSpec> sources = actual.getArgument("skills_sources");
        assertThat(sources).hasSize(1);
        // The original descriptor is not modified.
        assertThat((Object) chatModel.getArgument("skills_sources")).isNull();
    }

    @Test
    @DisplayName("Skills-object constructor keeps the descriptor language")
    public void testSkillsObjectKeepsDescriptorLanguage() {
        ResourceDescriptor chatModel =
                PythonResourceDescriptor.Builder.newBuilder("my_module.MyChatModel")
                        .addInitialArgument("skills", List.of("a"))
                        .build();
        Skills skills = Skills.fromLocalDir("/tmp/skill-b");

        ReActAgent agent = new ReActAgent(chatModel, null, null, skills);
        ResourceDescriptor actual = chatModelOf(agent);

        assertThat(actual)
                .isInstanceOf(PythonResourceDescriptor.class)
                .extracting(ResourceDescriptor::getModule, ResourceDescriptor::getClazz)
                .containsExactly("my_module", "MyChatModel");
        @SuppressWarnings("unchecked")
        List<SkillSourceSpec> sources = actual.getArgument("skills_sources");
        assertThat(sources).hasSize(1);
    }

    /** A class with no members at all, which Jackson refuses to render rather than rendering. */
    public static class FieldLess {}

    /** A member whose type carries no serializable state, so it renders to an empty object. */
    public static class WithCallback {
        public int count;
        public Function<String, String> callback;
    }

    /** A member that renders to a concrete type. */
    public static class WithCount {
        public int count;
    }

    /** A class reachable from itself, which the generator recurses on until the stack is gone. */
    public static class SelfReferential {
        public String name;
        public SelfReferential next;
    }
}
