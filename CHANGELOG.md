# Changelog

What is new in each version, in plain words. Newest first.

## [Unreleased]

- **Picking GGUF files of a repo is smarter and downloads only what is new.** The model file is always loaded first, so an importance-matrix file (imatrix) can no longer be mistaken for the model (it is ignored and never downloaded). Picking another model file replaces the old pick instead of adding to it, while its picture projector (mmproj) stays picked and the parts of a split model go together. A download fetches only the files that are missing: a projector another pick of the same repo already holds is linked, and adding a projector to an already downloaded model no longer needs a full re-download.
- **Self-check passes now see the pictures.** The model's check of its own rewrite was run without the pictures, so it could doubt a character's look it could not see; it now gets the same pictures as the rewrite itself.
- **A rewrite with Thinking off now returns only the rewritten prompt.** A Gemma 4 prompt without thinking now ends with an empty, closed thought block (as Google's prompt-format guide advises), so the model answers at once instead of reasoning aloud into the result. The cleaner also cuts a "Rewritten prompt:" or "Final output:" label in front and a closing "Note:" or "Explanation:" paragraph after the prompt.
- **A long generation error no longer takes over the screen.** The message is cut to its start and its end (the cause is usually at the end); the full text stays in ComfyUI's log.
- **SA-Solver no longer fails a generation when the GPU's CUDA runtime compiler is incomplete.** Its small coefficient table needs a function PyTorch compiles for the GPU on first use; if that compile fails (libnvrtc-builtins missing), the table is made on the CPU instead and the log says so.
- **The Shortcuts tab keeps its place and fills the Composer window.** Saving, deleting or importing a shortcut no longer jumps the list back to the top, the search field stays pinned while the list scrolls, and the list is as long as the Composer window allows instead of a small fixed box.
- **The Helper's checklist sits under its own message instead of being pinned at the top.** Each request keeps its own checklist: open and updating while the request works, folded once it is finished, and kept in the history.
- **A long bulk shortcut job no longer stalls after the first batch.** After each allowed shortcut edit the tool answer says how many shortcuts were changed in this request, how many have not been touched yet, and names the next 20, so the model can send the next batch straight away.
- **Optional Helper comments in the chat.** A new switch "Helper comments in the chat" (Settings > Editor, under the Helper switch, off by default) shows what the Helper writes between its actions as grey lines in the chat, instead of only inside its thinking line.
- **A model that gets stuck repeating itself gets a second chance.** When the engine cuts a round for repeating the same text, that round is thrown away and asked again (up to twice) with a new seed, with thinking off, and with an app note saying the last attempt got stuck and it should decide and call the tool at once.
- **The Helper adds up its shortcut changes.** The many "Changed N shortcuts" lines of one request are shown as one running total.
- **The Helper has its own temperature, 0.3 by default** (Composer > Enhance > Writing > "Helper temperature"). Tool calls need steady wording, so the Helper no longer uses the Enhance temperature, which can stay creative. The other Enhance sampling settings still apply to the Helper.
- **Helper thoughts stay folded unless you open them.** A thought you opened folds again when its run finishes, and sending a new message folds every thought (finished turns were kept as drawn, so an opened one used to stay open).
- **A bulk shortcut rename no longer fails because some of its names were already renamed.** The Helper reads the shortcut library fresh at the start of every request and keeps it current after each change you allow. A rename in a batch whose shortcut is already renamed (earlier in the request, or already carrying the new name) is now skipped and reported as done, instead of failing the whole batch with "no shortcut called".
- **The Helper is told plainly what a shortcut's name is and what its triggers are.** The rule and the shortcut tools now say a shortcut has a name (its label), triggers (the words that fire it) and replacements, that renaming means only new_name, and that triggers must not be touched to rename.
- **The Helper is told plainly what a shortcut's name is and what its triggers are.** The rule and the shortcut tools now say a shortcut has a name (its label), triggers (the words that fire it) and replacements, that renaming means only new_name, and that triggers must not be touched to rename.
- **The Helper's system prompt is shorter and lists every tool.** It now starts with a one-line list of all the tools (built from the tools themselves, so it never goes stale), keeps only the facts about how a project works, and adds a worked example plus two plain rules: never say a change worked before the tool confirms it, and never repeat a call that failed twice. The checklist tool is described as needed for every change.
- **A model stuck repeating itself is stopped, and long shortcut batches are split.** The GGUF engine ends a round once the end of its text is the same piece written four times over (it said why in the log) instead of filling the window. The Helper's bulk shortcut edit now takes at most 20 changes per call and tells the model to send long jobs as several calls, because very long tool calls are where small models lose track.
- **The Helper's pinned checklist is a plain fold again, without a box around it.** The extra card and the darker band behind it are gone; a pinned strip inside a window now takes the window's own colour.
- **The Helper's reminders now come from the app, not from you.** Every reminder the app sends the model (the checklist, "call the tool now", tool results) is marked as an app note, and the Helper's system prompt says such notes are not user messages, so the model no longer believes you wrote them.
- **The Helper no longer sees the Chat tab conversation.** It can read it only through a new read_chat_log tool, and only when you explicitly ask it to look at that chat log.
- **Clear in the Helper now forgets the conversation on the server too**, not only in the window: its saved turns and its finished request are dropped, and anything still running is stopped. The history still survives a refresh until you press Clear.
- **An empty model answer is no longer left empty.** When the model's very first token is its end-of-turn, the same round is asked again with another seed (up to twice), and the engine log now says why a round wrote nothing (the model ended its turn at once, or it was cancelled by a Stop or by the page going away).
- **A Helper request keeps running when the page is refreshed or loses its connection, and the Helper window follows it again.** The request now runs on the server; opening the Helper after a refresh shows what it has written so far and carries on live, including any Allow question still waiting. A dropped connection is followed again by itself instead of ending with "The answer stopped before it finished".
- **The Helper keeps its checklist pinned at the top in a collapsible card and updates it as items are checked**, and the working line names what it is doing, for example "Editing shortcuts… 10s" or "Reading prompts… 10s", or "Waiting for your answer".
- **The Helper window no longer slows the page down while the model writes.** Redraws are spaced out to about eight a second and only the turn that is still working is drawn again; before, every streamed piece rebuilt the whole history, which could freeze the page and its buttons.
- **A tool call the model wrote in a slightly different format now works.** Gemma texts in plain double or single quotes, or with no quotes (with spaces, brackets or commas in a shortcut name), are read. A call that still cannot be read is answered with an error telling the model so, instead of silently vanishing, which left the Helper in a loop with nothing happening.
- **The Helper's thinking line opens and closes reliably while it works.** The window no longer redraws between a click's press and release.
- **The Helper works every change request from a checklist and may not stop early.** Even for one change it writes a numbered plan (the app asks for it if it did not), does the items, and ticks each one (an item can only be ticked after a change was made, or closed with a reason). While items are open, the app sends the model back to them, up to 40 times, and stops only when every item is checked or it makes no headway three times in a row; then the answer says which items are still open. The progress shows as "Checklist: 3 of 10 done".
- **The Helper has its own "Max tokens in Helper" setting, unlimited by default**, so the rewrite's limit no longer cuts its answers short.
- **The Helper finds a shortcut even when it writes the name a little differently**: other case or spacing, a look-alike character, or a description in brackets after the name. A name it still cannot find comes back with the closest names, and a batch names every missing one at once and changes nothing.
- **The Helper keeps its conversation across a page refresh, with a Clear button.** The last 30 exchanges are kept in this browser for each project; Clear forgets them.
- **Text a model writes along the way no longer shows up as a Helper message.** Only its final answer is a message; anything written before or between tool calls sits in the folded thinking line.
- **Shortcut lookup in the Helper says how many there are and pages on.** It reports "N of the library's M shortcuts match, showing a to b" and how to get the next page, an empty search lists them all, and words in quotes are searched as an exact phrase, so a request for all shortcuts is not cut short unnoticed.
- **A Helper message that interrupts a change request now carries it on.** When the new message only adds to the stopped request ("Also no 3D look.") and the model answers in words, the Helper now makes the call as well. The same goes for any answer that announces a change without making it.
- **The Helper can edit, rename and delete many shortcuts in one go.** A new batch tool (edit_shortcuts) takes up to 60 changes and asks once for the whole batch; the question lists every change. Looking shortcuts up now returns up to 40 matches.
- The built-in intro above shortcut and lore entries is reworded: they are a dictionary used only as reference, never output verbatim, and their names never trigger scenes and are not written out. The version used while "Name shortcuts to the model" is off says the same without mentioning names.
- **"Name shortcuts to the model" switch** in the Enhance settings. Off, the model gets only the text of each shortcut or lore entry, never its name, so it has nothing to write into the prompt. On by default.
- The auto-hide timeline opens only as tall as its lanes need, with no empty space under them.
- The Helper no longer changes a rule it has not read. A rule edit sent blind is refused and answered with the rule's whole text, an edit that would cut most of a rule away is refused, and the rule name is accepted under other argument names (name, rule, prompt) and spellings. The forced round now repeats up to three times until the change is made.
- **The Helper has a Stop button, and a new message interrupts it.** Stop ends the writing and declines any change still waiting for Allow. A message sent while it works stops that work and is read together with the stopped request, which stays in the conversation as unanswered.
- **Three thinking switches: "Enable thinking in Helper", "Enable thinking on comments" and "Enable thinking on chat messages"** (Composer > Enhance, all on by default). With the main Thinking switch off, only those three think (and Gemma still reasons when it writes the conversation notes). A rewrite with nothing to follow uses the plain Thinking switch, so it can skip reasoning and save time.
- **Layout modes no longer turn the explorer off on half-precision models.** The packed-tensor unpack step turned a plain bf16 attention input into float32, which failed the first step and made the explorer quarantine itself.
- **Layout modes use less memory and accept packed (MXFP8) tensors.** The push now works in place instead of making several full-size copies on the first step, which could push the model out of the card, and a packed attention input is unpacked first.
- **The Helper is pushed into acting.** When a request asks for a change and the answer only talked about it, or only looked at a rule, the Helper runs one more round that names the fitting tools; a follow-up such as "will you correct them?" after an announcement counts as that request. It also lists only the tools that fit the request (all of them when the words name none), and no longer says "Done" after only looking something up.
- **Attention explorer can try layout modes (off by default).** A few fixed small pushes on the attention input of chosen blocks (3-45 by default), on the steps you name (the first by default), picked and learned from your ratings. Idea after CosPert.
- **The enhancer no longer writes shortcut names, labels or trigger words into the prompt.** The reference entries now say that those are only for telling the entries apart.
- **A dropdown stays open while the Composer refreshes in the background.** The Enhance tab redrew itself every time the engine reported progress (for example while the chat was being answered), which closed an open list; a list being used now holds the redraw back like a text field does.
- **The newest request is labelled "NEW REQUEST:" when earlier turns are sent as real turns.** A model could mistake an earlier request in its history for the current one; the last turn now says plainly that it is the one to answer.
- **Recurrent and sliding-window GGUF models reuse the conversation start too.** Qwen3.5 and Gemma cannot drop the end of their memory, so every chat or Helper turn used to read the whole prompt again. A copy of the memory is now kept at the start of the last turn, and the next request of the same conversation continues from it (about four times less waiting on a long prompt in a test).
- **Sliding-window GGUF models (Gemma) write faster on long prompts.** The memory no longer keeps a full-size copy for the sliding layers, which slowed every token as the prompt grew; such models read their prompt again from empty when it changes.
- **A crash inside llama.cpp now leaves a trace.** What llama.cpp says natively is kept in `models/funui_llm/llama_native.log`, and when the next load finds that the last ComfyUI session ended abruptly with llama.cpp loaded, the Log shows its last lines.
- **The tokens-per-second figure of GGUF models counts the writing only.** Before, the time to read the prompt was inside it, which could halve the number on a long prompt (the Helper's, a cleared memory). The Log now also says how long the prompt took to read, whether llama.cpp runs on the graphics card, and warns when a card is present but the installed llama.cpp build cannot use it.
- **The Helper knows which system prompt is which.** "The system prompt" now means the main rewrite instructions unless another one is named, and it is told the names the app shows for each rule.
- **One proper queue in front of the Enhance model.** Rewrites, Chat answers and Helper requests are served first come, first served, and one that has to wait says so ("waiting for the model: 1 request ahead"). Loading or unloading the model now waits for the request that is writing instead of pulling the model out from under it, and a request whose page went away no longer holds up the ones behind it.
- **The Chat tab no longer freezes a reply halfway.** An answer that finished while the next message was being typed stayed cut off until the page was reloaded; it now lands, and the caret stays in the message. A chat asked while a rewrite is still writing no longer takes over or ends the rewrite's live view.
- **The Helper can read and change the enhancer's system prompts**, with the same before-and-after Allow as in the Chat tab, and knows the macros they can use (such as the aspect ratio). When a request ends without any tool call, its answer now says that nothing was changed instead of reading as done.
- **The Helper's thinking is folded by default.** While the model works the window shows one line with a changing phrase and the seconds since the request began; a click opens the thinking as plain text, and afterwards it reads "Thought for 12s".
- **A scene the Helper adds starts with the first scene's references and aspect source**, so nobody has to attach them by hand.
- **The Composer's tabs are real tabs and stay pinned.** They look like folder tabs joined to the sheet, and only the sheet under them scrolls, on every tab (the Global prompt and Build tabs used to cut their content off without scrolling). New kit pieces: `tabs.folder` and a `scroll` option for `region.stack`.
- **A picture-reading model is now shown several pictures at once.** Prompt enhancer: the first four picture references of a scene go to a GGUF model with its projector file (before, only the first). Enhance nodes get `image_2` to `image_4` inputs for them.
- **The chat and rewrite system prompts now say that earlier turns are history.** The model is told that earlier requests, answers and (for the Helper) tool results are already done: it answers only the newest message and trusts the project as shown now.
- **The Helper no longer reports a failure when it made a change and then said nothing.** The window now says it is done, instead of "the model returned nothing".
- **The Helper reads tool calls that small models write sloppily.** Arguments with single or curly quotes, quotes or line breaks inside the text, trailing commas, or code fences are now understood, and Qwen 3.5's `<function=...><parameter=...>` call format works too. Before, such a call ended in a JSON error and nothing changed.
- **Recurrent and hybrid GGUF models (such as Qwen3.5) no longer fail on follow-up requests.** Their memory cannot drop only the end of a cached prompt, which made llama.cpp stop with "find_slot: non-consecutive token position". These models (found by their architecture, and by whether llama.cpp can shift their memory) now start from empty memory when the new prompt does not simply continue the last one.
- **The Helper tries again in text mode whenever a model answers without calling a tool**, not only when its answer announces a change. Plain questions ending with a question mark are just answered.
- **Models split into parts need only one part named.** For the Enhance engine, writing `owner/name::model-00002-of-00004.safetensors` (or a `-of-` GGUF part) now downloads every part, plus the index file and the config for safetensors parts, and loads them as one model. Before, only the named file came down and the model would not load. A missing part is named in the error.
- **The "Story" is now called "Global prompt" everywhere in the app**: the Composer tab and box, the cut words hint, the tour and the shortcut helpers. It is the same box as before.
- **A scene can have its own prefix.** In the Properties panel, a scene has an "Own prefix" switch: its text goes in front of that scene's prompt instead of the project's anchor, and an empty one means no prefix for that scene. Generate, the prompt preview and the Build tab all use it. The Helper can set it too.
- **The buttons at the right end of the timeline always come in the same order.** Helper first, then the others, then Sampler, and Composer last at the right edge, whichever order the features were loaded in.
- **New Helper window.** A "✦ Helper" button next to Composer opens a window in the middle of the screen with a message box and its own history. The model reads the project (every scene with its length and prompt) and can change a scene's prompt or length, add a scene, give a scene the same reference pictures as another ("same reference"; the scene is picked afterwards, so the media bin shows them), give every scene the same references, see the references pinned to the whole project, know the cuts (the Story's cut words, and which scenes are halves of one clip), rewrite the whole Story (all scenes at once), cut a scene in two, change the cut words, or change the anchor (also called the prefix), the postfix and its switch, negative prompt, $variables and shortcuts, pick model files and change any pipeline input (steps, sampler), apply a pipeline starting point, or change any setting of any module by name ("correct the blocks for the attention explorer", "turn on auto contrast"). Each change is shown first and only happens after you press Allow; when you refuse one, the helper asks itself what it got wrong and offers a corrected change or asks you a short question; it is a normal project edit, so Undo works. The window shows the helper's thinking as it is written (when thinking is on in the Enhance tab) and follows the answer down (unless you scrolled up to read). The helper treats every part of the Story between cut words as a separate prompt and changes only the scenes you mean, and writes the Story (the global prompt) as a quick summary, one short prompt per shot, without shot labels. If the model only talks about a change instead of calling the tool, it is told once to call it. It can also start the app's own actions for you, such as Generate, Generate selected, Render, Undo and Redo, after picking the scenes you name. It is off until you switch it on in Settings > Editor, which first shows a risk warning: how exactly a change matches your request depends heavily on the chosen model, and nobody guarantees correct instruction following. It can read the Chat tab conversation and the Notes (read only), so "do what I said in the chat" works. It works with any engine model: a model with tool calls uses them, and one without (or one that only talks about changes) writes its changes as blocks in its answer, which the app runs the same way, with the same Allow questions.
- **The Composer window keeps a margin at the bottom.** When its content was long enough to scroll, the last line sat flush against the window's edge in some browsers.
- **The Chat tab keeps every message across restarts**, and the model keeps its memory of the conversation too. The history used to live in memory only and hold just the last 20 runs, so older messages vanished and a restart emptied the tab. It is now saved in `enhancer_history.json` next to the pack (up to 1000 runs) and read again at start.
- **New macro `[[ASPECT_RATIO]]`** for every system prompt in Composer > Enhance: the clip's target aspect ratio, such as "16:9 landscape (1920 x 1080)", read from the scene's aspect ratio source picture (else the project's size). Pictures and videos in the media bin now remember their size for it. It says "(not known)" when the app did not send it.
- **New macro `[[VIDEO_LENGTH]]`** for every system prompt in Composer > Enhance: the estimated length of the clip being made, such as "5.04 seconds (121 frames at 24 fps)", taken from the scene's frames and frame rate (clips joined into one count together). In the Chat tab it is the length of the selected scene's clip. It says "(not known)" when the app did not send it.
- **Chat answers are no longer cut in the middle of a sentence.** An answer was cut at the first mention of words like "earlier in this conversation", "user feedback" or "original prompt:", which a model uses in plain speech. Now only a heading at the start of a line counts as an echoed label.
- Enhancer: the engine status names a model short (repo and file, or repo and how many files) instead of the whole source text, and a finished download is a separate sentence after the loaded model's line. The download line says what is being downloaded.
- **H3 corrector** (Settings > Engine > Sampling, MiniMax H3, experimental): a small trained network adds a fix to the picture at every step, capped at a share of the prediction. Put a corrector file (for example Sulphur 3's `s3_corrector.safetensors`) in `models/h3_correctors`, switch it on and pick the file; strength and the largest fix are adjustable. Needs no extra ComfyUI nodes. Costs one extra pass per step. Tried on a real H3 model with the Sulphur 3 corrector.
- Enhancer: a rule change the model writes at the end of its answer is now found even when the model forgets the closing line, so the Allow window opens (a model that ignored the closing line used to change nothing and show no window).
- Enhancer: pressing Download now refreshes the Downloaded models list right away (a quick download used to be missed until the page was reopened), warns when a download or load is already running, and marks files in the repo list that are already downloaded.
- Pressing Stop while a prompt is being rewritten by the built-in engine now stops the rewrite right away and ends the run, instead of waiting for the model to finish.
- **No token limit by default.** Max tokens is now 0 = no limit: the engine writes until the model stops by itself or the context window is full (a model's reasoning no longer counts against it either). Set a number to cap it. ComfyUI's own text generation still uses 4096 when it is 0.
- **Download and Load are separate buttons** (Composer > Enhance). Download fetches only the checked files of a model (a new quantization, say) and leaves the loaded model alone; the result shows in Downloaded models, where Use loads it. Load now loads into memory, downloading first only when files are missing.
- **A thinking model's answer is no longer cut off.** Its reasoning had to fit in the same token limit as the answer, so a long reasoning left the answer short or empty. The engine now gives the reasoning its own 3000 tokens on top of the length setting.
- **Conversation rules say the last message is always the prompt to rewrite**, even when it repeats an earlier request word for word. A thinking model had read such a repeat as "no new prompt" and been unsure what to do.
- **The model can propose a rule change by writing it, not only by calling the tool.** When it cannot or does not call the edit tool, it writes the new rule at the end of its answer; you get the same before/after window, and the answer shows "Rule change ...: saved". The rules also tell it to propose a change when it notices one (the same correction repeated, a rule that keeps failing), not only when asked.
- **GGUF: bigger window.** llama.cpp now gets a 32k window (was 16k) and the reply length is cut to what fits after the request, with a clear message when the request itself does not fit. The CUDA libraries of the Python environment are only put in front when llama.cpp cannot load without them.
- **Composer tabs stay pinned** at the top of the window while a sheet scrolls.
- **GGUF: "failed to find a memory slot" fixed.** llama.cpp keeps a full-size cache for the model's sliding-window layers so the unchanged start of a request can be reused, and a request that still fails to decode is run again once from an empty memory.
- **"Guess ahead" is off by default.** It only helps when a rewrite mostly repeats the prompt and slows creative rewrites down.
- **llama.cpp installs with visible progress and mends itself.** The engine line shows what pip is doing while llama-cpp-python installs. The first time, it installs llama-cpp-python together with the CUDA 12 runtime packages (for a graphics card), says up front that a restart of ComfyUI may be needed, and asks for that restart if the fresh install does not load in the running process. It loads the CUDA libraries already in your Python environment first. Only if it still fails after the restart does it fall back to the processor build. The Enhance tab now also points to GGUF as the fast choice.
- **GGUF models can see the picture.** When the model's `mmproj` file sits beside it (check it with the model in the repo list), "Use the picture" sends the scene's picture to the model through it. A model without a projector says in the run status that the picture was not sent.
- **The engine runs .gguf models with llama.cpp.** Name a Hugging Face repo and a file (`owner/name::model-Q4_K_M.gguf`) or a local .gguf path. GGUF models keep their quantized weights, which is much lighter and faster than widening them, and thinking, streaming and the model editing its own rules work as with other models. The llama-cpp-python package is installed automatically the first time a .gguf is loaded (a build for your graphics card when there is one); if that fails, the error gives the command to run by hand.
- **Several files of one repo can be downloaded together.** In Composer > Enhance, a repo's weight files are listed as checkboxes: check every part of a split model (or one fp8 file) and only those are downloaded. Nothing checked downloads the whole repo.
- **Output speed under the model's answers.** The Chat tab shows "Average output speed: N tokens/sec" below each answer the engine wrote.
- **"Use ComfyUI's text generation instead"** (Composer > Enhance, off by default). On, the enhancer and the Chat tab write with ComfyUI's own text generation as before, and FunUI's engine stays off.
- **The chat follows the model as it writes.** While an answer streams in, the Chat tab scrolls down with it, unless you scrolled up to read.
- **The engine says how it runs a model.** The line under the model picker now ends with "bf16" or "fp8". fp8 widens every layer at every word, which is slow; bf16 (the automatic choice when the graphics card has room) does not.
- **Engine models whose tokenizer is missing now load.** A ComfyUI text-encoder file carries its own tokenizer, which the engine uses when the downloaded config folder has none of the usual kind. Models that only ship an older sentencepiece tokenizer file need the `sentencepiece` package, now in the requirements (restart after updating); the error says so.
- **Downloaded engine models can be picked again.** Each model in Composer > Enhance > Downloaded models has a Use button next to Delete: it sets the engine model and starts loading it.
- **Speed controls and live writing for the enhancer.** Composer > Enhance > Speed: Precision (a single fp8 model file can be widened to bf16 once when the graphics card has room, so every word is faster; the default picks this on its own), Reuse the unchanged start of a request, and Guess ahead (0 to 10 words checked in one step). Every model output (rewrites, self-checks, summaries, chat answers) is shown live: a run that happens during Generate now appears in the Enhance tab ("Writing now") and in the Chat tab, thinking and answer as they form. The loaded model stays in memory between Generates and reloads only when the model or precision changes.
- **Pick the engine model from ComfyUI's list, or one file of a Hugging Face repo.** Composer > Enhance > "From ComfyUI's list" runs a text encoder file from your models list on the engine; its config and tokenizer are downloaded once from the Hugging Face model of the same name (or the one you add as `@owner/name`). "File in the repo" downloads a single weights file, for example the fp8 one, instead of every copy. This replaces the separate Encoder and Encoder family choices: a file picked there runs on the engine, and a project that saved an encoder keeps working. A single-file fp8 model whose output layer shares its input table (such as Gemma 4 12B) now loads.
- **Fixes from a review.** A shortcut pair like `DANCE|ZOOM` stays plain text outside Prompt markup (LoRA training captions no longer carry internal marks). A weight inside one option, as in `[(red:1.3) car|truck]`, is applied again. Step merging keeps one random second seed for both stages of a run. Choosing a different engine model now loads it, and closing the chat while it writes stops the writing.
- **Self-check passes for rewrites** (Composer > Enhance > Self-check passes, 0 to 3, off by default). After the model writes a rewrite it checks that rewrite against its own rules and the original prompt, says what is weak, and writes an improved version. A pass that gives nothing usable leaves the earlier version. Each pass is another model call, so a rewrite takes longer. The wording is editable under System prompts > Self-check.
- **The chat model can edit its own system prompts, with your approval** (Composer > Enhance > "Let the model edit its own rules", needs an Engine model that supports tool calls). Ask it in the Chat tab to change how it works and it proposes a new text for one of its rules. A window titled "Allow changes to system prompt?" shows which prompt and a before/after diff, with Allow and Disallow. Allowed changes are saved into the project and apply from the next request; after a refusal the model is told and does not push the same change again. The wording of this rule is editable like the others.
- **The enhancer's engine is faster on repeated requests.** The start of a request that matches the previous one (the long instructions, the earlier turns) is not read again, and short stretches the model is about to repeat from the prompt are checked in one pass. The words written are the same as without them. The run's status shows tokens per second and how much of the prompt was reused. A chat sent while a rewrite runs waits for its turn on the engine and does not depend on ComfyUI's queue.
- **The enhancer's engine can run a single-file fp8 model.** Name a Hugging Face repo that holds one fp8 text-encoder file (the ComfyUI layout) and the engine downloads that file plus its base model's settings, keeps the weights in fp8 (about half the memory of bf16) and runs them with the normal transformers model code. `owner/name::file.safetensors@owner/base` names the file and base by hand.
- **In "Only my own text" mode the model is now told what the global prefix and postfix say.** They still are not part of the text it rewrites and are put back exactly as written, but the instructions now carry them as fixed context, so the model writes for the whole prompt. Editable under Composer > Enhance > System prompts > Fixed text around it. `[[PROMPT_PREFIX]]` and `[[PROMPT_POSTFIX]]` are new macros.
- **The prompt enhancer shows exactly what the model was given.** The Enhance tab's "Sent to the model" now appears for every rewrite, with the instructions, the earlier turns, your comments, ratings and notes as the model received them. A macro with nothing behind it yet says so plainly ("nothing yet: none was given so far") so the model does not read it as missing context.
- **The prompt enhancer can run on its own engine** (Composer > Enhance > Engine model). Name any Hugging Face model that the transformers library supports (or a folder) and it is downloaded once and run directly, instead of through ComfyUI's text encoder. Chat answers appear as they are written, with the model's thinking shown first, so there is no waiting for the whole answer or refreshing the Chat tab. Rewrites during Generate can use it too. Thinking and tool-call formats are recognised per model family.
- **The Enhance tab is organised by what depends on what.** Settings follow the switch they depend on and are greyed out, not hidden, while that switch is off. Every instruction the model is told, including the rewrite instructions, the reference intro and the rating line opening, now sits in one "System prompts" section in the order a request meets them.
- **The Composer's Story box scales with its window.** It takes the height the window has left, so the Variables section and the hint under it stay in view when the window or the screen is smaller, instead of being pushed out of sight.
- **Macros in the enhancer's instructions.** Write `[[PREVIOUS_MODEL_OUTPUT]]`, `[[LAST_USER_COMMENT]]`, `[[LAST_CHAT_MESSAGE]]`, `[[LAST_RATING]]`, `[[COMMENTS_HISTORY]]` and more in Instructions or any system prompt, and each is replaced by what the conversation holds when the model is asked. The full list is in Composer > Enhance > System prompts > Macros.
- **Every instruction the prompt enhancer gives the model can be edited** (Composer > Enhance > "System prompts"): chat answers, the Notes rule, the conversation notes writer, how comments are applied, the conversation rules (for plain encoders and for Gemma 4) and the markup rule. A field left empty uses the built-in text, which is shown in the field.
- **Chat tab messages keep their order.** Every message and comment has a number that only rises, so a comment stays exactly where it was written, even when the rewrite it was about is gone from the enhancer's memory (it keeps the last 20) or the enhancer restarted.
- **Better summaries of long enhancer conversations.** The model that shortens a long conversation now gets each scene labelled (your request, your comments, the rating, its own answer) and is asked for notes under clear headings: scenes, your feedback and ratings, and what still applies. With Gemma 4 it thinks first and samples with Google's own settings (temperature 1.0, top_p 0.95, top_k 64, min_p 0.05, no repetition penalty); other encoders write the notes plainly and without a repetition penalty.
- **An edited prompt no longer follows its old version in the enhancer's conversation** (with "Send everything into chat"). When you generate a scene again, the enhancer's earlier request and answer for that scene are replaced instead of kept, so the new rewrite is not pulled back toward the old prompt.
- **The model's thinking shows in the Chat tab** (when "Thinking" is on), as a message of its own just before each answer, for rewrites and instant replies. It no longer shows in the Enhance tab.
- **Options a version no longer has never stop a run.** A saved pipeline that still holds a setting a node no longer has is run without it, and a saved modifier choice that is no longer one of the options uses that setting's default. The Log says what was ignored or replaced. Values you typed that are out of range are still refused.
- **A "Comment..." button next to a scene's rating** (Composer > Enhance > "Comment button"). It opens a small window to comment on the enhancer's last rewrite, the same as a comment in the Chat tab. Off by default.
- **Instant replies in the Chat tab** (with "Send everything into chat" and an Encoder of its own picked in Enhance). "Send" asks the model now and shows its answer as a bubble; the model answers under a chat prompt of its own, with the whole conversation so far. "Comment" keeps your note instead and it goes with the next prompt as part of one message. A reply waits while a generation is running.
- **Notes the model can edit** (Composer > Chat > Notes). Standing instructions, one per line, that every rewrite and chat reply starts with. When you tell the model something lasting in the Chat tab, it can add a note or drop one by number, and the Chat shows what it noted; you can edit the notes yourself at any time.
- **The Chat tab looks like a chat:** your messages in bubbles on the right, the model's on the left, in the model's own words (without the global prefix and postfix; the full prompt is in the Enhance tab). With "Send everything into chat" on, every request and answer of the project's conversation shows there.
- **The prompt enhancer can keep one running conversation** (Composer > Enhance > "Send everything into chat"). With it ticked, each request goes to the model together with the earlier requests of the project and the answers it gave (the last 8 turns), instead of as a one-time request, so the model keeps the characters and style it has already written. Every comment you make (Chat tab or Comment button) belongs to the conversation, whatever prompt it was made on, and is sent once, with the next request, then stays in its place in the history (Gemma 4 gets the earlier turns as real chat turns; other encoders get them as one block of text). When a request would nearly fill the model's context window (32768 tokens, or the size you set under "Context window"; a smaller model's own size wins), the conversation so far is summarised and the summary is sent instead of every turn; "Summarise long conversations" turns that off. The Chat tab shows how full the context window is and says where the conversation was summarised, with the summary text to read; a notice also appears when it happens. "Refresh" there reloads the conversation, and the tab opens at the newest message ("Top ^" goes back to the start). "Reset" there starts a new conversation. The conversation is kept in memory only and is gone when ComfyUI restarts. Off by default.
- **The prompt enhancer can pass the rating of the previous scene** (Composer > Enhance > "Pass the rating", available with "Send everything into chat"). With it ticked, one closing line is added to the end of the request: the rating of the last rated scene before this one ("This is the user's rating of the video made from the previous prompt: Liked"), or "This prompt was not rated." when its rating was forgotten. "Rating line opening" replaces the built-in words. Nothing is added when no earlier scene is rated and there is no comment. Off by default.
- **The prompt enhancer can look at the previous scene's last frame** (Composer > Enhance > "Pass the last frame of previous scene as enhancer input", shown while Continuity is on). With it ticked, the model is shown that picture instead of the first reference. The first scene, or a shot with no carried frame, uses the first reference and the Log says so. Off by default.
- **The prompt enhancer accepts the app's own shortcuts file as a reference file** (Composer > Enhance > Lorebook / shortcut files). It was refused with "not a lorebook or shortcuts file".
- **The prompt enhancer can rewrite only your own text** (Composer > Enhance > Rewrites). "Only my own text" sends the model what sits between the global prefix and postfix and puts the prefix and postfix back exactly as written; "The whole prompt" (default) rewrites everything. In both, `(word:1.5)` weights and `[a|b]` alternatives are kept as written; if the model changes them, the prompt runs unchanged and the Log says so.
- **Long variable values scroll inside their box** instead of growing or being cut off, and each box remembers where your cursor and scroll were when you move between variables or leave and come back to the Variables list.
- **The scene name above the timeline is now a menu.** Click "Scene 1" next to Takes and Upscale to see every scene in the project and jump to any of them, the same as clicking its clip on the timeline.
- **Continuity can carry the end of the previous scene, not only its last picture** (Project settings, experimental). Tick "Pass the end of the previous scene as a video reference" to pass its last frames with their own sound, "Pass the sound of the previous scene's end as a sound reference" to pass the sound alone, or both. The carry-over length is set in seconds or in frames; changing one updates the other at the frame rate the clips are generated at. The frames are taken as generated, before any upscale or interpolation (if that file is gone, the finished video is used and the Log says so). Off by default.
- Experimental: a still-picture VAE (H3-Image) can be picked as the decode-only VAE for H3 video. Each latent frame is decoded as a picture and each picture is held to fill the clip length, so frames stay sharp but motion is steppy. Untested on a real model.
- **DynaShift** (Settings > sampling, experimental): bends the noise levels of the chosen scheduler with a shift that moves from a start value to an end value across the run, Two stages included. Made for turbo and few-step runs; full effect up to 8 steps, fading out on longer runs. No extra model calls.
- **Two stages can use another sampler and scheduler for the full-size stage** (Settings > Engine > Sampling). "Same as the first stage" is the default and changes nothing. A different scheduler spaces the last steps its own way and joins the first stage at the hand-over.
- **The previous DynaShift is now called EchoShift** (it steers away from clips you disliked). Its settings and banked ratings are kept.
- **MiniMax H3-Image** (experimental): a new "Start from" pipeline, MiniMax H3-Image · Text to Image and Edit. Write a prompt for a picture, or add reference pictures and refer to them as `<Picture 1>`, `<Picture 2>`, ... in the prompt (the first is the picture to edit; up to 16). The result is a PNG picture, not a clip: the preview and the timeline show it as a picture, it exports like any clip (one second per picture), and Length and FPS are not offered for it (both are set to 1). REINS and the Attention explorer work on it (the explorer leaves out the sound and clip regions, which it does not have); Seed explorer and Decisiveness are offered too. You need the model, its VAE and its text encoder files; the model's licence excludes the EU, the UK, South Korea and the United States.
- **Noise merging is now called Step merging**, because most of its modes merge the model's answers at a step, not noise. Settings are kept.
- **Alternating prompts** (Settings > general): write `[dance|zoom]` in a prompt, or `DANCE|ZOOM` between two shortcuts, for two options, A and B. "Main prompt reads" sets what the prompt says by default: "A while B" (default), "A B", only A or only B. Switch on "Read A and B separately" (off by default) to also assign steps: "Steps that read only A" and "Steps that read only B" give each step you name its own option, with no extra model pass. A `|` next to anything that is not a shortcut is left as typed. Two options per bracket.
- **Simultaneous actions** (Step merging, needs "Read A and B separately"): with `[A|B]` in the prompt, each merged step is answered once on A and once on B from the same picture, and the two answers are mixed at your strength. "Second pass new noise" sets how much of the noise B starts from is new (0 keeps the same noise for both). Strength 0 reads only A, and strength 1 with new noise 1 reads only B, each with a single pass. Steps named in Step merging win over the per-step lists. Costs one extra model pass per merged step.
- **Fixed: Alternating prompts did nothing in H3 pipelines saved earlier** (the "[A|B]" in the prompt was read as having no options). Such a pipeline is now linked to the settings when it is opened.
- **Removed: timed phrases `[phrase@2-4]`, and `[a|b]` as a blend of two prompts.** `[a|b]` now means two actions at once (above). `(word:1.5)` weights work as before.
- **Reference sheet** (Assets panel, "Reference sheet..."): pick rows and columns, a picture and a label for each slot, and a title. You get one white sheet with the pictures in a grid, each labelled below and the title in the top left corner, saved to the media bin as a picture. Pictures come from the bin or an upload; each is cropped from the middle to the slot's shape.
- **Everything you generate now goes to ComfyUI's temp folder, and nothing to its output folder.** Clips, pictures, upscaled and interpolated videos are removed when ComfyUI restarts; a project opened afterwards shows those clips as not made yet, and keeps their ratings (shortcut scores and Clear saved ratings still count them). The same happens in an open project when you delete generated files or ComfyUI restarts: the app checks after a delete in Settings > Generated & Output Files, when ComfyUI is back, and before every Generate, so a deleted clip is not played from the browser's memory and the next shot does not try to continue from it. To keep a result, use Export or "Save to media bin". Settings > Generated & Output Files lists what was generated since ComfyUI started.
- **Fixed: the app could show a deleted video instead of the new one.** After the output folder was emptied, ComfyUI numbered files from 00001 again, so a new video took a deleted video's name, and the player, the takes and the next shot's start picture kept using the deleted one. Each file name now has a short random part (`FunUI_00001_3fa94c1e.mp4`) and is never used twice, upscaled and interpolated videos included, and the browser no longer keeps a preview by its file name alone.
- **Fixed: generating again with a fixed seed after deleting the video** gave back the deleted file's name instead of a video. The video is now saved again.

## [5.2.0] "Classy Chopper" - 2026-11-08

### New: finishing touches for the finished video

- **Watermark.** Put a text or a picture from your media bin in any corner of the finished video (Settings > Engine > Post). You choose how see-through it is, how big, how far from the edge, and can add an outline. Transparent PNGs keep their transparency. Off by default.
- **Picture clean-up options** (Settings > Engine > Post, all off by default): smooth jagged edges, auto contrast, auto brightness and colour strength. They are measured once for the whole clip, so nothing flickers.
- **Interpolate** adds in-between frames to a finished video with NVIDIA's frame generation (2 to 4 times the frames, same length and sound). Needs an RTX card. Choose Never, a button next to the rating, or automatic after every render (Settings > Editor).
- **Save Video quality is now a setting, and the default is "high".** Files are bigger and take a little longer to save; colours are also very slightly brighter than before.

### New: faster and sharper generating

- **Two stages** (Settings > Engine > Sampling, MiniMax H3): the first steps run on a smaller picture and the rest finish at full size. You can pick how the picture is enlarged in between, including a learned upscaler file.
- **Veda sparse attention** (experimental, NVIDIA GPU): a quicker way to compute attention. The model file can be downloaded from Settings > Models.
- **MXFP8 for the H3 loader** (Blackwell GPUs): roughly twice the matrix speed.
- **Decode-only VAE** in the VAE Loader: a second, faster VAE used only for the last decoding step (for example the Sol-H3 VAE).
- **Resolution rule** in Project settings: choose whether Width and Height mean the short side, the long side or the pixel count. New projects start on the short side; older projects keep their size.
- **Run meter** (off by default) shows how grainy the picture is and how much it still changes each step, and compares it with your last run of the same seed. Handy to see what one setting did.

### New: LoRAs, prompts and shortcuts

- **LoRA Loader holds up to 8 LoRAs** and shows one slot at a time. Each can be applied to the first stage, the second stage or both (when Two stages is on).
- **More LoRA files work**, including the Diffusers-style files such as NVIDIA's Open-H3-Regenerate-2K refiner.
- **Prompt enhancer: Dynamic style** asks for a clear action, a change over time and secondary motion in every shot.
- **Shortcuts: Find and replace, and Common phrases.** Search every shortcut at once, preview the changes, then apply them. Common phrases lists wording shared by several shortcuts so you can fix it in one place.
- **Attention explorer: "Change the sound too"**. Turn it off to leave the sound's attention alone.

### Changed

- **Simpler scene and project settings.** These are gone because the timeline already does the job: "Pause after this clip" (drag the clip instead; edges snap to neighbours and the playhead, Shift holds it free), "Source trim" (trim with the clip's edges), the scene's "Source" choice and "Start shots" (a scene now starts from what it has: a prompt, a picture or references).
- **Video input is set once per project.** The separate per-scene copy is gone.
- **Continuity moved to Project settings** as "Pass the last frame of previous scene as reference into next scene". A shot after a clip that ends dark always starts fresh.
- **Auto Montage is gone.** Projects that already contain its pieces open normally; the pieces stay as ordinary clips.
- **"Video detail" is gone** from Post settings. Projects that used it open normally and the Log mentions it once.
- **Rarely used settings** (Video input, final render size, split preview) sit under "More".
- **Scenes that used the old Source choices** now carry on from the previous clip like the default.

### Fixed

- **Upscaling no longer makes the next generation reload all models.** Upscaling now runs beside ComfyUI instead of as one of its queued jobs. Generate is refused while an upscale model is working, so the two cannot fight over memory.
- **The Model Loader no longer reloads the model on every run** when two loaders use the same file, and a switch of attention or SLA settings frees the earlier copy cleanly.
- **No more false "memory leak" warning** from ComfyUI after a model's links were lost.
- **No more "Should have ta>=t0" warning** with SDE-type samplers. Pictures were never affected.
- **Reference numbers on media tiles match the model's inputs**, counted per kind (the first sound is R1 even if a picture holds the first slot). Pinned references go to the input of their own number, and the Log says when something had nowhere to go.
- **LoRA training says why** when it cannot copy the model, and learns from LoRA files that name their parts `lora_down` and `lora_up`.
- **The timeline is never shorter than its lanes**, the header fits one row from 1024px wide, and the guided tour follows buttons that get renamed.
- **"Build" keeps weaker drafts in play** and no longer treats the word "constructor" as a weight.
- **Stop works at once** while an Upscale or Interpolate waits for its turn.
- **Settings of removed features** are cleared when a project opens, so the Log stops warning about them.

<details>
<summary>Technical notes</summary>

- Veda needs CUDA, Triton and a predictor file in `ComfyUI/models/veda`; with Veda the SLA sparsity, rising share and references settings are not read yet.
- Two stages hands the second stage the model's clean guess; the first stage keeps the full picture's shape within a few percent.
- Save Video "high" is constant quality 16 (it was 23) with a slower encoder preset, and the finished picture is rounded to 8 bits instead of truncated. Older saved pipelines get "high".
- MXFP8 needs torch 2.10 or newer with CUDA 13; elsewhere it is skipped with a note.
- Interpolate needs nvidia-vfx 0.2 or newer; switching the add-on on updates an older copy.

</details>

## [5.1.0] "Classy Chopper" - 2026-10-27

### New

- **LTX 2.3 is tested on a real model**, with two presets: Video to Video (direct) and Image to Video (from References). Both fall back to text to video when no picture or video is given, and the run says so.
- **Attention explorer** (needs a Taste key): each run tries a small attention change per group of blocks, and your ratings decide which changes stay. Works on MiniMax H3 and LTX.
- **Noise merging** can run a whole second video beside yours from a second seed and blend the two, or mix the starting noise of two seeds.
- **Identity bank** (LTX, needs a Taste key) keeps the look of clips you like and leans on it in the next ones. Not yet tried on a real LTX run.
- **Guide frames** (LTX) use a picture from the clip before as a gentle guide for the next shot.
- **Rising sparsity** for SLA: skip fewer attention blocks early and more late. One real H3 render kept its quality.
- **Video input** (Project and Scene tabs): trim, resize, crop and re-time reference videos before the model sees them. Off by default.
- **Temporal style: auto** picks a style for each shot from its own words.
- **Flip width and height** button in Project settings.
- **Block settings know the model**: they show how many blocks it has and keep your list inside that range.

### Changed

- **LTX: the start picture is now reference picture 1** in the scene's References selector instead of a picture dropped on the timeline. Image to Video and Video to Video are separate presets.
- **On LTX, only Noise merging, the Attention explorer and Taste are offered for now.** The other enhancements come back once they have been proven on a real LTX run. Your saved settings are kept.
- **Shot camera and SLA are shown for MiniMax H3 only.**
- **Long clips use less memory** when loaded.
- **The ComfyUI log survives a restart**, and the Log window has a "Previous run" button. It also notes RAM and GPU memory use, so a run that gets killed for memory leaves a trace.

### Fixed

- Clip names in Japanese (or any non-ASCII name) no longer fail a run on a console that is not UTF-8.
- Restoring a saved pipeline no longer adds another preset's input slots.
- Older LTX Image to Video pipelines fall back to text to video when no picture is given, instead of refusing to start.
- Cancelling during a Noise merging second pass now cancels.
- Attention explorer ratings made on H3 no longer teach LTX.
- Guide frames no longer crash on an invalid position.

## [5.0.3] "Classy Chopper" - 2026-10-25

### Changed

- **FunPack is now FunUI**, with a new logo. The 5.x codename is "Classy Chopper".
- **The app moved to `/fun/`.** The old address redirects.
- Picker wheel settings are saved with the project.
- The repository moved to `digital-garbage/FunUI`; the old address redirects, so existing copies keep updating.
- Projects, imported workflows and saved settings with old `FunPack` names are renamed as they load. Nodes already placed in ComfyUI's own graph under the old names need to be added again.
- Decisiveness is now a "Commitment" slider from -10 to +10 (0 is off); a value saved by an older version resets to 0.

### New (experimental)

- **PTD few-step** (MiniMax H3 only): renders in 4 to 8 steps using a PTD adapter file from `ComfyUI/models/ptd`. Ported from ComfyUI-PTD-Compact by Nynxz.

## [5.0.2] - 2026-10-22

### Changed

- **The interface follows Apple's Human Interface Guidelines more closely**: roomier windows, better text contrast, nothing under 11px, a clearer focus ring and selection marks, and sliders whose value you can type. The Composer and Training windows open with a Genie motion.
- New README screenshots and a clearer note on how the project is maintained.

### Fixed

- "Restarting ComfyUI..." no longer appears twice.

## [5.0.1] - 2026-10-21

Fixes from real H3 runs on rented GPUs, plus the first usable LoRA training.

### New

- **LoRA training on H3** with a step-by-step window (Data, Tags, Train, Results): previews at chosen steps, Pause and Resume, Restore and continue, live graphs and downloads.
- **Gallery** for saved videos and pictures, with preview, Select mode and confirmed delete.
- **15 reference slots for H3** (9 pictures, 3 videos, 3 sounds).
- **Log colours**: a healthy log is plain text; warnings are amber and errors red.
- A bypass switch per node in Models & Pipeline, and two new Quick wheel entries: Disable all enhancements and Fix the seed.
- The loading window names the module being loaded.
- Experimental: **Distilled Flow samplers** and **Sharpness**.

### Fixed

- **Save Video no longer loses a video** when the GPU encoder will not open; it saves on the CPU and says why.
- **Reference videos and sound** now reach H3's video and audio inputs, are no longer cut to 15 seconds, and a video's own sound comes along.
- The Render button was dark with near-black text.
- RTX Super Resolution says why it failed to load.
- Taste direction works on the CPU.
- Text-only H3 runs no longer need a picture input.
- The Log survives a crash.
- Media upload shows progress and no longer stalls the server on big clips.

## [5.0.0] "Cunning Cranberry" - 2026-10-16

A rewrite from scratch, not an update of 4.x. Same idea (a video studio inside ComfyUI that learns from your ratings), new design: every feature is a module that switches on or off by itself, and a feature that fails to load is left out while the run carries on. 4.x projects are not imported.

### What it is

- **A browser studio** with a timeline, Composer (story in one box, templates, variables, shortcuts), properties, media bin, preview, takes, rating, Export and Render, project wizard and guided tour.
- **Models & Pipeline**: the pipeline as cards. Pick files, swap or add nodes, import a ComfyUI workflow. It refuses what would not build and says why.
- **Modules** (Settings > Modules): switch each feature on or off.
- **Learning from ratings**: a Taste key remembers your ratings, and the features that use them start acting once there are enough.

### Models

- **MiniMax H3**: the model v5 is built and tested around.
- **LTX 2.x**: tested on a real model, including image to video from References.
- **Stable Diffusion 1.5**: experimental.

### Features

- **Prompt enhancer**: a language model rewrites each finished prompt (has its own encoder pick).
- **Auto Montage**: builds a new cut from your rendered clips; originals stay untouched.
- **Upscale** a finished render with an upscale model or NVIDIA RTX Super Resolution.
- **LoRA training** (add-on, experimental): Data, Tags, Train and Results steps, with previews made by your own pipeline.
- **Add-ons** (spaCy, nvidia-vfx, onnxruntime) install on the next start after you switch them on.
- Many sampling and guidance features, some proven and some experimental. See Settings > Modules.

### Not carried over

- 4.x projects and settings are not imported; start new projects in 5. ComfyUI workflow files can still be imported.
- Features that never proved out stay out.

### Known limits

- `.gguf` text encoders need the ComfyUI-GGUF node pack and only work for architectures it understands.
- On H3 the scene picture sets the video's shape only; it is not a start frame.
- Some modules switch the model compiler off for the run, which costs sampling time.

See [How to start generating](docs/how-to-start.md).
