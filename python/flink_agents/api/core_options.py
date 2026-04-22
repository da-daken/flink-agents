################################################################################
#  Licensed to the Apache Software Foundation (ASF) under one
#  or more contributor license agreements.  See the NOTICE file
#  distributed with this work for additional information
#  regarding copyright ownership.  The ASF licenses this file
#  to you under the Apache License, Version 2.0 (the
#  "License"); you may not use this file except in compliance
#  with the License.  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
# limitations under the License.
#################################################################################
import os
from enum import Enum
from typing import Any

from pyflink.java_gateway import get_gateway

from flink_agents.api.configuration import ConfigOption


def covert_j_option_to_python_option(j_option: Any) -> ConfigOption:
    """Convert a Java config option to a Python config option."""
    key = j_option.getKey()
    default = j_option.getDefaultValue()
    type_name = j_option.getTypeName()

    if type_name == "java.lang.String":
        config_type = str
    elif type_name == "java.lang.Integer":
        config_type = int
    elif type_name == "java.lang.Long":
        config_type = int
    elif type_name == "java.lang.Boolean":
        config_type = bool
    elif type_name == "java.lang.Float":
        config_type = float
    elif type_name == "java.lang.Double":
        config_type = float
    else:
        msg = f"Unsupported type: {type_name}"
        raise TypeError(msg)

    return ConfigOption(key, config_type, default)


class AgentConfigOptionsMeta(type):
    """Metaclass for FlinkAgentsCoreOptions."""

    def __init__(
        cls, name: str, bases: tuple[type, ...], attrs: dict[str, Any]
    ) -> None:
        """Initialize the metaclass for FlinkAgentsCoreOptions."""
        super().__init__(name, bases, attrs)

        jvm = get_gateway().jvm
        cls.jvm = jvm

    def __getattr__(cls, item: str) -> ConfigOption:
        j_option = getattr(
            cls.jvm.org.apache.flink.agents.api.configuration.AgentConfigOptions,
            item,
        )

        python_option = covert_j_option_to_python_option(j_option)
        return python_option


class ErrorHandlingStrategy(str, Enum):
    """Error handling strategy for Agent.

    Currently, only works for chat action.
    """

    RETRY = "retry"
    FAIL = "fail"
    IGNORE = "ignore"


class ShortTermMemoryStateTtlUpdateType(str, Enum):
    """TTL refresh policy for short-term memory MapState; matches Java Agent.ShortTermMemoryStateTtlUpdateType."""

    ON_READ_AND_WRITE = "ON_READ_AND_WRITE"
    ON_CREATE_AND_WRITE = "ON_CREATE_AND_WRITE"


class ShortTermMemoryStateTtlVisibility(str, Enum):
    """TTL read visibility for short-term memory MapState; matches Java Agent.ShortTermMemoryStateTtlVisibility."""

    NEVER_RETURN_EXPIRED = "NEVER_RETURN_EXPIRED"
    RETURN_EXPIRED_IF_NOT_CLEANED_UP = "RETURN_EXPIRED_IF_NOT_CLEANED_UP"


class AgentConfigOptions(metaclass=AgentConfigOptionsMeta):
    """CoreOptions to manage core configuration parameters for Flink Agents."""

    JOB_IDENTIFIER = ConfigOption(
        key="job-identifier",
        config_type=str,
        default=None,
    )


class AgentExecutionOptions:
    """Execution options for Flink Agents."""

    ERROR_HANDLING_STRATEGY = ConfigOption(
        key="error-handling-strategy",
        config_type=ErrorHandlingStrategy,
        default=ErrorHandlingStrategy.FAIL,
    )

    MAX_RETRIES = ConfigOption(
        key="max-retries",
        config_type=int,
        default=3,
    )

    NUM_ASYNC_THREADS = ConfigOption(
        key="num-async-threads",
        config_type=int,
        default=os.cpu_count() * 2,
    )

    CHAT_ASYNC = ConfigOption(
        key="chat.async",
        config_type=bool,
        default=True,
    )

    TOOL_CALL_ASYNC = ConfigOption(
        key="tool-call.async",
        config_type=bool,
        default=True,
    )

    RAG_ASYNC = ConfigOption(
        key="rag.async",
        config_type=bool,
        default=True,
    )

    SHORT_TERM_MEMORY_STATE_TTL_MS = ConfigOption(
        key="short-term-memory.state-ttl.ms",
        config_type=int,
        default=0,
    )

    SHORT_TERM_MEMORY_STATE_TTL_UPDATE_TYPE = ConfigOption(
        key="short-term-memory.state-ttl.update-type",
        config_type=ShortTermMemoryStateTtlUpdateType,
        default=ShortTermMemoryStateTtlUpdateType.ON_READ_AND_WRITE,
    )

    SHORT_TERM_MEMORY_STATE_TTL_VISIBILITY = ConfigOption(
        key="short-term-memory.state-ttl.visibility",
        config_type=ShortTermMemoryStateTtlVisibility,
        default=ShortTermMemoryStateTtlVisibility.NEVER_RETURN_EXPIRED,
    )
