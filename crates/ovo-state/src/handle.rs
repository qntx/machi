//! Actor-backed conversation handle (prompt index + usage ledger).

use ovo_types::{Message, OvoError, Role, Usage};
use tokio::sync::{mpsc, oneshot};

use crate::ledger::UsageLedger;
use crate::strict::{StrictAppendError, check_append};

/// Serializable conversation snapshot.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ChatStateSnapshot {
    /// Ordered messages.
    pub messages: Vec<Message>,
    /// Usage ledger (session + per-prompt + per-model).
    pub usage: UsageLedger,
    /// Message indices that start a user prompt / turn boundary.
    #[serde(default)]
    pub prompt_index: Vec<usize>,
}

enum Command {
    Append {
        message: Message,
        strict: bool,
        reply: oneshot::Sender<Result<(), OvoError>>,
    },
    Replace {
        messages: Vec<Message>,
        reply: oneshot::Sender<()>,
    },
    Restore {
        snapshot: ChatStateSnapshot,
        reply: oneshot::Sender<()>,
    },
    Snapshot {
        reply: oneshot::Sender<ChatStateSnapshot>,
    },
    RecordUsage {
        usage: Usage,
        subagent: bool,
        model: Option<String>,
        reply: oneshot::Sender<()>,
    },
    RecordCompaction {
        strategy: String,
        reply: oneshot::Sender<()>,
    },
    MarkIncomplete {
        reply: oneshot::Sender<()>,
    },
    Shutdown {
        reply: oneshot::Sender<()>,
    },
}

/// Cloneable handle to a single-writer conversation actor.
#[derive(Clone)]
pub struct ChatStateHandle {
    tx: mpsc::UnboundedSender<Command>,
}

impl std::fmt::Debug for ChatStateHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ChatStateHandle").finish_non_exhaustive()
    }
}

impl ChatStateHandle {
    /// Spawn an actor with optional seed messages.
    #[must_use]
    pub fn spawn(seed: Vec<Message>) -> Self {
        let (tx, rx) = mpsc::unbounded_channel();
        let prompt_index = prompt_index_from_messages(&seed);
        tokio::spawn(actor_loop(rx, seed, UsageLedger::new(), prompt_index));
        Self { tx }
    }

    /// Spawn an actor from a full checkpoint.
    #[must_use]
    pub fn spawn_from_snapshot(snapshot: ChatStateSnapshot) -> Self {
        let (tx, rx) = mpsc::unbounded_channel();
        let prompt_index = if snapshot.prompt_index.is_empty() {
            prompt_index_from_messages(&snapshot.messages)
        } else {
            snapshot.prompt_index
        };
        tokio::spawn(actor_loop(
            rx,
            snapshot.messages,
            snapshot.usage,
            prompt_index,
        ));
        Self { tx }
    }

    /// Append a message (strict tool pairing enforced).
    ///
    /// # Errors
    ///
    /// Returns invariant or actor-channel failures.
    pub async fn append(&self, message: Message) -> Result<(), OvoError> {
        self.append_inner(message, true).await
    }

    /// Append without strict pairing (escape hatch for repair paths).
    ///
    /// # Errors
    ///
    /// Channel failures only.
    pub async fn append_unchecked(&self, message: Message) -> Result<(), OvoError> {
        self.append_inner(message, false).await
    }

    async fn append_inner(&self, message: Message, strict: bool) -> Result<(), OvoError> {
        let (reply, rx) = oneshot::channel();
        self.tx
            .send(Command::Append {
                message,
                strict,
                reply,
            })
            .map_err(|_| actor_gone())?;
        rx.await.map_err(|_| actor_gone())?
    }

    /// Replace full history (compaction). Does not clear usage.
    pub async fn replace(&self, messages: Vec<Message>) {
        let (reply, rx) = oneshot::channel();
        if self.tx.send(Command::Replace { messages, reply }).is_ok() {
            let _ = rx.await;
        }
    }

    /// Restore messages **and** usage from a checkpoint.
    pub async fn restore(&self, snapshot: ChatStateSnapshot) {
        let (reply, rx) = oneshot::channel();
        if self.tx.send(Command::Restore { snapshot, reply }).is_ok() {
            let _ = rx.await;
        }
    }

    /// Snapshot messages + usage + prompt index.
    ///
    /// # Errors
    ///
    /// Actor channel closed or reply dropped (handle is unusable).
    pub async fn snapshot(&self) -> Result<ChatStateSnapshot, OvoError> {
        let (reply, rx) = oneshot::channel();
        self.tx
            .send(Command::Snapshot { reply })
            .map_err(|_| actor_gone())?;
        rx.await.map_err(|_| actor_gone())
    }

    /// Messages only (convenience).
    ///
    /// # Errors
    ///
    /// Same as [`Self::snapshot`].
    pub async fn messages(&self) -> Result<Vec<Message>, OvoError> {
        Ok(self.snapshot().await?.messages)
    }

    /// Turn-boundary indices (user messages that open a prompt).
    ///
    /// # Errors
    ///
    /// Same as [`Self::snapshot`].
    pub async fn prompt_index(&self) -> Result<Vec<usize>, OvoError> {
        Ok(self.snapshot().await?.prompt_index)
    }

    /// Message count.
    ///
    /// # Errors
    ///
    /// Same as [`Self::snapshot`].
    pub async fn len(&self) -> Result<usize, OvoError> {
        Ok(self.messages().await?.len())
    }

    /// True when conversation is empty.
    ///
    /// # Errors
    ///
    /// Same as [`Self::snapshot`].
    pub async fn is_empty(&self) -> Result<bool, OvoError> {
        Ok(self.len().await? == 0)
    }

    /// Usage ledger snapshot.
    ///
    /// # Errors
    ///
    /// Same as [`Self::snapshot`].
    pub async fn usage(&self) -> Result<UsageLedger, OvoError> {
        Ok(self.snapshot().await?.usage)
    }

    /// Persist via a [`crate::persistence::ChatPersistence`] backend.
    ///
    /// # Errors
    ///
    /// Actor channel or backend I/O failures.
    pub async fn save_to(
        &self,
        store: &dyn crate::persistence::ChatPersistence,
    ) -> Result<(), OvoError> {
        let snap = self.snapshot().await?;
        store.save(&snap).await
    }

    /// Replace state from a persistence backend when present.
    ///
    /// # Errors
    ///
    /// Backend I/O failures.
    pub async fn load_from(
        &self,
        store: &dyn crate::persistence::ChatPersistence,
    ) -> Result<bool, OvoError> {
        match store.load().await? {
            Some(snap) => {
                self.restore(snap).await;
                Ok(true)
            }
            None => Ok(false),
        }
    }

    /// Open a handle: load checkpoint when present, else empty seed.
    ///
    /// # Errors
    ///
    /// Backend I/O failures.
    pub async fn open_or_new(
        store: &dyn crate::persistence::ChatPersistence,
    ) -> Result<Self, OvoError> {
        match store.load().await? {
            Some(snap) => Ok(Self::spawn_from_snapshot(snap)),
            None => Ok(Self::spawn(vec![])),
        }
    }

    /// Record main-loop usage (current prompt + session main).
    pub async fn record_main_usage(&self, usage: Usage) {
        self.record_usage(usage, false, None).await;
    }

    /// Record main-loop usage with model attribution.
    pub async fn record_main_usage_model(&self, usage: Usage, model: impl Into<String>) {
        self.record_usage(usage, false, Some(model.into())).await;
    }

    /// Record nested agent usage.
    pub async fn record_subagent_usage(&self, usage: Usage) {
        self.record_usage(usage, true, None).await;
    }

    async fn record_usage(&self, usage: Usage, subagent: bool, model: Option<String>) {
        let (reply, rx) = oneshot::channel();
        if self
            .tx
            .send(Command::RecordUsage {
                usage,
                subagent,
                model,
                reply,
            })
            .is_ok()
        {
            let _ = rx.await;
        }
    }

    /// Record a compaction event at the current message length.
    pub async fn record_compaction_at(&self, strategy: impl Into<String>) {
        let (reply, rx) = oneshot::channel();
        if self
            .tx
            .send(Command::RecordCompaction {
                strategy: strategy.into(),
                reply,
            })
            .is_ok()
        {
            let _ = rx.await;
        }
    }

    /// Mark usage incomplete.
    pub async fn mark_incomplete(&self) {
        let (reply, rx) = oneshot::channel();
        if self.tx.send(Command::MarkIncomplete { reply }).is_ok() {
            let _ = rx.await;
        }
    }

    /// Stop the actor.
    pub async fn shutdown(self) {
        let (reply, rx) = oneshot::channel();
        if self.tx.send(Command::Shutdown { reply }).is_ok() {
            let _ = rx.await;
        }
    }
}

fn prompt_index_from_messages(messages: &[Message]) -> Vec<usize> {
    messages
        .iter()
        .enumerate()
        .filter_map(|(i, m)| (m.role == Role::User).then_some(i))
        .collect()
}

fn actor_gone() -> OvoError {
    OvoError::new(
        ovo_types::ErrorCode::StatePersistence,
        "chat state actor is gone",
    )
}

fn map_strict(err: StrictAppendError) -> OvoError {
    OvoError::new(ovo_types::ErrorCode::StateInvariant, err.to_string())
}

async fn actor_loop(
    mut rx: mpsc::UnboundedReceiver<Command>,
    mut messages: Vec<Message>,
    mut usage: UsageLedger,
    mut prompt_index: Vec<usize>,
) {
    while let Some(cmd) = rx.recv().await {
        match cmd {
            Command::Append {
                message,
                strict,
                reply,
            } => {
                let result = if strict {
                    match check_append(&messages, &message) {
                        Ok(()) => {
                            if message.role == Role::User {
                                prompt_index.push(messages.len());
                            }
                            messages.push(message);
                            Ok(())
                        }
                        Err(e) => Err(map_strict(e)),
                    }
                } else {
                    if message.role == Role::User {
                        prompt_index.push(messages.len());
                    }
                    messages.push(message);
                    Ok(())
                };
                let _ = reply.send(result);
            }
            Command::Replace {
                messages: next,
                reply,
            } => {
                messages = next;
                prompt_index = prompt_index_from_messages(&messages);
                let _ = reply.send(());
            }
            Command::Restore { snapshot, reply } => {
                messages = snapshot.messages;
                usage = snapshot.usage;
                prompt_index = if snapshot.prompt_index.is_empty() {
                    prompt_index_from_messages(&messages)
                } else {
                    snapshot.prompt_index
                };
                let _ = reply.send(());
            }
            Command::Snapshot { reply } => {
                let _ = reply.send(ChatStateSnapshot {
                    messages: messages.clone(),
                    usage: usage.clone(),
                    prompt_index: prompt_index.clone(),
                });
            }
            Command::RecordUsage {
                usage: u,
                subagent,
                model,
                reply,
            } => {
                if subagent {
                    usage.record_subagent(u);
                } else {
                    usage.record_main(u);
                    let prompt_i = prompt_index.len().saturating_sub(1);
                    usage.record_prompt(prompt_i, u);
                }
                if let Some(m) = model {
                    usage.record_model(m, u);
                }
                let _ = reply.send(());
            }
            Command::RecordCompaction { strategy, reply } => {
                usage.record_compaction_at(messages.len(), strategy);
                let _ = reply.send(());
            }
            Command::MarkIncomplete { reply } => {
                usage.mark_incomplete();
                let _ = reply.send(());
            }
            Command::Shutdown { reply } => {
                let _ = reply.send(());
                break;
            }
        }
    }
}

#[cfg(test)]
#[allow(clippy::expect_used, reason = "unit tests")]
mod tests {
    use super::*;

    #[tokio::test]
    async fn prompt_index_tracks_user_turns() {
        let h = ChatStateHandle::spawn(vec![Message::system("s")]);
        h.append(Message::user("a")).await.expect("a");
        h.append(Message::assistant("b")).await.expect("b");
        h.append(Message::user("c")).await.expect("c");
        let idx = h.prompt_index().await.expect("idx");
        assert_eq!(idx, vec![1, 3]);
        h.record_main_usage_model(Usage::new(2, 1), "mock").await;
        let u = h.usage().await.expect("usage");
        assert_eq!(u.per_prompt.len(), 2);
        assert!(u.per_model.contains_key("mock"));
        h.record_compaction_at("max_messages").await;
        assert_eq!(h.usage().await.expect("usage2").compaction_at.len(), 1);
        h.shutdown().await;
    }
}
