"""Phase 2 communication bus interface — scaffold only.

In Phase 1, the bus is not instantiated.  ``BaseTrafficEnv.connect_communication_bus()``
accepts any object implementing this interface, so Phase 2 extensions
will not require any change to the environment code.

Phase 2 implementation notes:
    - Implement ``RedisMessageBus`` (or ``ZMQMessageBus``) using this interface.
    - Each intersection env publishes its observation at each step.
    - Neighbour agents subscribe to adjacent intersection states.
    - ``PPOAgent`` reads neighbour states via ``receive()`` before predicting.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class AbstractMessageBus(ABC):
    """Interface for inter-agent message passing.

    All communication back-ends (in-process, Redis, ZMQ, etc.) must
    implement this protocol.  Kept minimal to keep Phase 2 flexible.
    """

    @abstractmethod
    def publish(self, agent_id: str, message: Dict[str, Any]) -> None:
        """Broadcast a message from agent ``agent_id`` to all subscribers.

        Args:
            agent_id: Unique identifier of the publishing agent.
            message:  Arbitrary JSON-serialisable payload (e.g. observation).
        """
        ...

    @abstractmethod
    def receive(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve the most recent message from a neighbour agent.

        Args:
            agent_id: ID of the agent to receive from.

        Returns:
            Most recent message payload, or ``None`` if nothing available.
        """
        ...

    @abstractmethod
    def register(self, agent_id: str) -> None:
        """Register an agent with the bus before it can publish or receive.

        Args:
            agent_id: Unique agent identifier.
        """
        ...


class NoOpMessageBus(AbstractMessageBus):
    """Phase 1 placeholder — all operations are no-ops.

    Used to satisfy the ``connect_communication_bus()`` API without
    any actual inter-process communication overhead.
    """

    def publish(self, agent_id: str, message: Dict[str, Any]) -> None:
        """No-op in Phase 1.

        Args:
            agent_id: Ignored.
            message:  Ignored.
        """

    def receive(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Always returns None in Phase 1.

        Args:
            agent_id: Ignored.

        Returns:
            None.
        """
        return None

    def register(self, agent_id: str) -> None:
        """No-op in Phase 1.

        Args:
            agent_id: Ignored.
        """
