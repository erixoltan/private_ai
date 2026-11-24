# GLM 5.3 on Amazon Bedrock

## Initial Prompt

I am trying to understand data security in Amazon Bedrock. I have a number of references.

GLM 5.3, a frontier coding model, is available on amazon bedrock

https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-zai-glm-5-3.html#model-card-zai-glm-5-3-pricing

Security, Guardrails and Observability

https://docs.aws.amazon.com/bedrock/latest/userguide/security.html

Shared Responsibility Model

https://aws.amazon.com/compliance/shared-responsibility-model/

Security

https://docs.aws.amazon.com/bedrock/latest/userguide/security-overview.html

Data Protection

https://docs.aws.amazon.com/bedrock/latest/userguide/data-protection.html

What is the danger to our proprietary data when we use GLM 5.3 on Amazon Bedrock? Will the data be retained? Will it be used in training? Will there be a danger of a data breach where our proprietary data could leak out? What are the possibilities that our proprietary code or private customer data will be compromised? What would need to happen for it to be compromised?

## Analysis

*Researched Nov 4, 2026 from AWS's public documentation and terms. These are AWS's published statements, not terms we have negotiated. The AWS Service Terms are the binding version, and AWS can change them.*

### Summary

AWS's documents support both claims that matter to us: GLM 5.3 on Bedrock does not train on our data, and by default it does not retain our data. Z.ai, the company that makes GLM 5.3, never receives our data. The realistic ways our code or data could leak are on our side: credentials, settings and the coding agent. Breaches at AWS or Z.ai are the less likely route.

### Will our data be used for training?

No.

- The Bedrock FAQ says: *"AWS and the third-party model providers will not use any inputs to or outputs from Amazon Bedrock to train"* models. It also says *"your content is not used to improve the base models and is not shared with any model providers."*
- AWS Service Terms §50.3 lets some AI services use customer content to improve the service. It does not list Bedrock. §50.2 says the output we generate *"is Your Content."*
- Z.ai has no access. AWS copies the model into a Model Deployment Account that the Bedrock service team owns and runs. The data-protection page says model providers *"don't have access to Amazon Bedrock logs or to customer prompts and completions."*

### Will our data be retained?

By default, mostly no. There are four exceptions, and each one is a setting we control.

- **AWS abuse-detection storage does not apply to GLM 5.3.** AWS keeps inputs and outputs for up to 30 days only for certain OpenAI models and for Claude Fable 5 and 5.1. GLM 5.3 is not on that list, and its model card marks "Abuse detection" as not supported.
- **Saved responses (the catch).** GLM 5.3 supports OpenAI's Responses API. In Bedrock's `default` retention mode, that API saves each response unless the request says not to (`store=true` is the default). **Fix:** set the account's data retention mode to `none` in each Region we use. That blocks saved responses and refuses any model that requires retention. The docs show an organization policy (SCP) that stops anyone from changing the mode.
- **Our own invocation logging.** Model invocation logging records full prompts and responses to our own S3 or CloudWatch. It is off by default. If we turn it on, those logs hold copies of our code and must be protected like source code.
- **Prompt cache.** For GLM 5.3, cached prompt prefixes are kept for at least 30 minutes on the server that handled the request. The pages we read do not say whether the cache is isolated per customer account. Ask AWS.

**How to verify:** with retention mode set to `none`, Bedrock refuses calls to any model that requires retention. If GLM 5.3 calls still succeed under `none`, GLM 5.3 does not require retention.

### Where does our data travel?

GLM 5.3 is available only through cross-Region inference. It cannot be pinned to a single Region.

| Profile | Where requests are processed | Notes |
| --- | --- | --- |
| US (`us.zai.glm-5.3`) | US Regions only (us-east-1, us-east-2, us-west-1, us-west-2) | Use this one |
| Global (`global.zai.glm-5.3`) | Any of about 30 commercial Regions worldwide, including Tokyo, Mumbai, São Paulo and Tel Aviv | About 10% cheaper; block it with an SCP |

With either profile, traffic stays on the AWS network and is encrypted in transit. If any data is retained, it is stored in the Region that processed the request. CloudTrail records the processing Region in `additionalEventData.inferenceRegion`.

### What would have to happen for our code or customer data to leak?

Most likely first:

1. **Stolen credentials.** The getting-started guide uses long-term Bedrock API keys. Anyone holding one could read any saved responses or invocation logs and run up charges. Use short-lived IAM roles instead, require MFA, and keep keys off developer machines and VMs.
2. **The coding agent does something harmful.** The model only returns text, but an agent turns that text into commands on our machine. A malicious instruction hidden in a file the agent reads (prompt injection), or a behavior trained into the model, could make it run a command that sends files elsewhere or writes insecure code. The defense is our rule that agents get less access than people, plus blocking outbound network access from the agent's VM except to Bedrock and GitHub. This risk is the same whoever hosts the model.
3. **Misconfiguration.** Examples: retention mode left at `default`, a log bucket left readable, someone using the global profile, or invocation logging turned on without anyone noticing.
4. **A failure at AWS.** This means a breach of the Bedrock service or an AWS insider. AWS describes Bedrock as using a zero operator access (ZOA) model. Bedrock is in scope for SOC 1, 2 and 3, ISO 27001, FedRAMP Moderate and HIPAA eligibility, and processing data with it falls under the AWS Data Processing Addendum (Service Terms §1.14). This is the same trust we already place in AWS for everything else in our account.
5. **Legal compulsion.** Like any US provider, AWS can be compelled to hand over data. If retention is set to `none`, there is little to hand over.

Z.ai is not on this list. It has no access to the deployment accounts, the logs, or our prompts and completions.

### Customer-licensed data and protected domains

AWS's terms are strong, but AWS still operates the service. Our rule that customer-licensed data never reaches a model outside our control still applies. Until legal reviews the customer contracts, Class R material, including customer data, trade-secret code and protected domains, stays off Bedrock. Classes C, D and P are good candidates once the Board approves Bedrock.

### Implications for the policy draft

- **Bedrock on-demand is Zone 3, not Zone 2.** The second draft says a service running in our own AWS account "may belong in Zone 2." On-demand Bedrock runs in an AWS-owned Model Deployment Account, not in ours. It is an enterprise vendor contract, so it belongs in Zone 3. Zone 2 would mean running the GLM 5.3 open weights ourselves, for example on our own EC2 or SageMaker instances. That sentence, and the matching open question, should be corrected when this is folded in.
- **Proposed required settings if the Board approves Bedrock for Zone 3:**
  - Retention mode `none` in every Region we use, locked by an SCP.
  - US geographic profile only, with the global profile denied by an SCP.
  - Invocation logging off, or on only with an approved, access-controlled bucket.
  - Short-lived IAM roles instead of long-term API keys.
  - CloudTrail on, so every request and its processing Region are recorded.
  - Outbound network access from agent VMs limited to approved endpoints.

### Questions for the AWS account team

1. Access to GLM 5.3 is limited to eligible customers. Are we eligible, and what is the process?
2. Does GLM 5.3 accept retention mode `none`? This can be confirmed by testing, but we want it in writing.
3. Is the prompt cache isolated per customer account?
4. Does the account-level retention setting in the source Region govern requests that cross-Region inference processes in other US Regions?

### Sources

- [GLM 5.3 model card](https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-zai-glm-5-3.html)
- [Security, Guardrails and Observability](https://docs.aws.amazon.com/bedrock/latest/userguide/security.html)
- [Security overview](https://docs.aws.amazon.com/bedrock/latest/userguide/security-overview.html)
- [Data protection](https://docs.aws.amazon.com/bedrock/latest/userguide/data-protection.html)
- [Data retention](https://docs.aws.amazon.com/bedrock/latest/userguide/data-retention.html)
- [Abuse detection](https://docs.aws.amazon.com/bedrock/latest/userguide/abuse-detection.html)
- [Cross-Region inference](https://docs.aws.amazon.com/bedrock/latest/userguide/cross-region-inference.html)
- [Prompt caching](https://docs.aws.amazon.com/bedrock/latest/userguide/prompt-caching.html)
- [Model invocation logging](https://docs.aws.amazon.com/bedrock/latest/userguide/model-invocation-logging.html)
- [Amazon Bedrock FAQs](https://aws.amazon.com/bedrock/faqs/)
- [AWS Service Terms](https://aws.amazon.com/service-terms/) (Bedrock: §50.12; AI Services: §50.2–50.3; data protection: §1.14)
- [AWS Shared Responsibility Model](https://aws.amazon.com/compliance/shared-responsibility-model/)
